/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 * 
 * Copyright (C) 2021
 * ELKI Development Team
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package elki.clustering.kmeans;

import java.util.List;

import elki.clustering.kmeans.LloydKMeans.Instance;
import elki.clustering.kmeans.initialization.KMeansInitialization;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.KMeansModel;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDs;
import elki.database.relation.Relation;
import elki.distance.NumberVectorDistance;
import elki.distance.UnitLengthEuclidianDistance;
import elki.logging.Logging;
import elki.utilities.optionhandling.parameterization.Parameterization;

/**
 * The unoptimized spherical k-means algorithm based on the work of Lloyd and
 * Forgy
 * (independently). This version utilizes a distance Function that uses the dot
 * product for vectors of unit length. For more information, see
 * {@link UnitLengthEuclidianDistance}
 * <p>
 * Reference:
 * <p>
 * S. Lloyd<br>
 * Least squares quantization in PCM<br>
 * IEEE Transactions on Information Theory 28 (2)<br>
 * previously published as Bell Telephone Laboratories Paper
 * <p>
 * E. W. Forgy<br>
 * Cluster analysis of multivariate data: efficiency versus interpretability of
 * classifications<br>
 * Abstract published in Biometrics 21(3)
 *
 * @author Alexander Voﬂ
 *
 * @navassoc - - - KMeansModel
 *
 * @param <V> vector datatype
 */
public class LloydUnoptimizedSphericalKMeans<V extends NumberVector> extends LloydKMeans<V> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(LloydUnoptimizedSphericalKMeans.class);

  protected boolean varstat = true;

  public LloydUnoptimizedSphericalKMeans(int k, int maxiter, KMeansInitialization initializer, boolean varstat) {
    super(UnitLengthEuclidianDistance.STATIC, k, maxiter, initializer);
    this.varstat = varstat;
  }

  @Override
  public Clustering<KMeansModel> run(Relation<V> relation) {
    Instance instance = new Instance(relation, distance, initialMeans(relation));
    instance.run(maxiter);
    return instance.buildResult(true, relation);
  }

  protected static class Instance extends LloydKMeans.Instance {

    public Instance(Relation<? extends NumberVector> relation, NumberVectorDistance<?> df, double[][] means) {
      super(relation, df, means);
    }

    @Override
    protected Logging getLogger() {
      return LOG;
    }

    @Override
    protected int iterate(int iteration) {
      long start = System.currentTimeMillis();
      means = iteration == 1 ? means : means(clusters, means, relation);
      LOG.info("means : " + (System.currentTimeMillis() - start));
      start = System.currentTimeMillis();
      int assign = assignToNearestCluster();
      LOG.info("assign : " + (System.currentTimeMillis() - start));
      return assign;
    }

    /**
     * Returns the mean vectors of the given clusters in the given database.
     *
     * @param clusters the clusters to compute the means
     * @param means the recent means
     * @param relation the database containing the vectors
     * @return the mean vectors of the given clusters in the given database
     */
    protected static double[][] means(List<? extends DBIDs> clusters, double[][] means, Relation<? extends NumberVector> relation) {
      final int k = means.length;
      double[][] newMeans = new double[k][];
      for(int i = 0; i < k; i++) {
        DBIDs list = clusters.get(i);
        if(list.isEmpty()) {
          // Keep degenerated means as-is for now.
          newMeans[i] = means[i];
          continue;
        }
        DBIDIter iter = list.iter();
        double[] sum = relation.get(iter).toArray();
        // Add remaining vectors (sparse):
        for(iter.advance(); iter.valid(); iter.advance()) {
          plusEquals(sum, relation.get(iter));
        }
        // normalize to unit length
        newMeans[i] = UnitLengthEuclidianDistance.STATIC.normalize(sum);
      }
      return newMeans;
    }

  }

  /**
   * Parameterization class.
   *
   * @author Alexander Voﬂ
   */
  public static class Par<V extends NumberVector> extends AbstractKMeans.Par<V> {
    @Override
    public void configure(Parameterization config) {
      super.configure(config);
      super.getParameterVarstat(config);
    }

    @Override
    public LloydUnoptimizedSphericalKMeans<V> make() {
      return new LloydUnoptimizedSphericalKMeans<V>(k, maxiter, initializer, varstat);
    }
  }
}
