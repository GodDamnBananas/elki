package experimentalcode.remigius.Visualizers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.batik.util.SVGConstants;
import org.w3c.dom.Element;

import de.lmu.ifi.dbs.elki.algorithm.clustering.ByLabelClustering;
import de.lmu.ifi.dbs.elki.data.Clustering;
import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.data.cluster.Cluster;
import de.lmu.ifi.dbs.elki.data.model.Model;
import de.lmu.ifi.dbs.elki.database.Database;
import de.lmu.ifi.dbs.elki.logging.LoggingUtil;
import de.lmu.ifi.dbs.elki.math.AggregatingHistogram;
import de.lmu.ifi.dbs.elki.result.Result;
import de.lmu.ifi.dbs.elki.result.ResultUtil;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.Flag;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionID;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.ParameterException;
import de.lmu.ifi.dbs.elki.visualization.css.CSSClass;
import de.lmu.ifi.dbs.elki.visualization.css.CSSClassManager.CSSNamingConflict;
import de.lmu.ifi.dbs.elki.visualization.svg.SVGPlot;
import experimentalcode.remigius.ShapeLibrary;

/**
 * Generates a SVG-Element containing a histogram representing the distribution of the database's objects.
 * 
 * @author Remigius Wojdanowski
 * 
 * @param <NV>
 * @param <N>
 */
public class HistogramVisualizer<NV extends NumberVector<NV, N>, N extends Number> extends ScalarVisualizer<NV, N> {
  
  public static final OptionID STYLE_ROW_ID = OptionID.getOrCreateOptionID("histogram.row", "Alternative style: Rows.");

  private final Flag STYLE_ROW_PARAM = new Flag(STYLE_ROW_ID);

  private boolean row;

  private static final String NAME = "Histograms";

  private static final int BINS = 20;

  private double frac;

  private Result result;

  private Clustering<Model> clustering;

  public HistogramVisualizer() {
    addOption(STYLE_ROW_PARAM);
  }

  @Override
  public List<String> setParameters(List<String> args) throws ParameterException {
    List<String> remainingParameters = super.setParameters(args);
    row = STYLE_ROW_PARAM.getValue();
    rememberParametersExcept(args, remainingParameters);
    return remainingParameters;
  }

  public void init(Database<NV> database, Result result) {
    init(database, NAME);
    this.frac = 1. / database.size();
    this.result = result;
    setupClustering();
  }

  private void setupClustering() {
    List<Clustering<?>> clusterings = ResultUtil.getClusteringResults(result);

    if(clusterings != null && clusterings.size() > 0) {
      clustering = (Clustering<Model>) clusterings.get(0);
    }
    else {
      clustering = new ByLabelClustering<NV>().run(database);
    }
  }

  // private void setupCSS() {
  //
  // int clusterID = 0;
  // for (Cluster<Model> cluster : clustering.getAllClusters()){
  // clusterID+=1;
  // CSSClass bin = visManager.createCSSClass(ShapeLibrary.BIN + clusterID);
  // bin.setStatement(SVGConstants.CSS_FILL_OPACITY_PROPERTY, "0.5");
  // bin.setStatement(SVGConstants.CSS_FILL_PROPERTY,
  // COLORS.getColor(clusterID));
  // visManager.registerCSSClass(bin);
  // }
  // }

  private void setupCSS(SVGPlot svgp, int clusterID) {
    CSSClass bin = new CSSClass(svgp, ShapeLibrary.BIN + clusterID);
    bin.setStatement(SVGConstants.CSS_FILL_OPACITY_PROPERTY, "0.5");
    bin.setStatement(SVGConstants.CSS_FILL_PROPERTY, COLORS.getColor(clusterID));

    try {
      svgp.getCSSClassManager().addClass(bin);
      svgp.updateStyleElement();
    }
    catch(CSSNamingConflict e) {
      LoggingUtil.exception("Equally-named CSSClass with different owner already exists", e);
    }
  }

  @Override
  public Element visualize(SVGPlot svgp) {
    Element layer = ShapeLibrary.createSVG(svgp.getDocument());

    Map<Integer, AggregatingHistogram<Double, Double>> hists = new HashMap<Integer, AggregatingHistogram<Double, Double>>();

    // Creating histograms
    int clusterID = 0;
    for(Cluster<Model> cluster : clustering.getAllClusters()) {
      clusterID += 1;
      AggregatingHistogram<Double, Double> hist = AggregatingHistogram.DoubleSumHistogram(BINS, scales[dim].getMin(), scales[dim].getMax());
      for(int id : cluster.getIDs()) {
        hist.aggregate(database.get(id).getValue(dim).doubleValue(), frac);
      }
      hists.put(clusterID, hist);
      setupCSS(svgp, clusterID);
    }

    // Visualizing
    // TODO: Drawing centered instead of left-end values of bins.
    for(int key = 1; key <= hists.size(); key++) {
      AggregatingHistogram<Double, Double> hist = hists.get(key);
      if(row) {
        for(int bin = 0; bin < hist.getNumBins(); bin++) {
          // TODO: calculating the value *must* be simpler. Something is wrong
          // here.
          double val = hist.get(bin * (scales[dim].getMax() - scales[dim].getMin()) / BINS);
          layer.appendChild(ShapeLibrary.createRow(svgp.getDocument(), getPositioned(bin * hist.getBinsize(), dim), 1 - val, hist.getBinsize(), val, dim, key, bin));
        }
      }
      else {
        // TODO: Path is still buggy.
        Element path = ShapeLibrary.createPath(svgp.getDocument(), 1, 1, key, Integer.toString(key));
        // Just a hack to ensure the path ends at its beginning.
        ShapeLibrary.addLine(path, 0, 1);
        for(int bin = 0; bin < hist.getNumBins(); bin++) {
          double val = hist.get((bin * (scales[dim].getMax() - scales[dim].getMin()) / BINS));
          ShapeLibrary.addLine(path, getPositioned(bin * hist.getBinsize(), dim), 1 - val);
        }
        layer.appendChild(path);
      }
    }
    return layer;
  }
}
