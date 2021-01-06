/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 *
 * Copyright (C) 2019
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

import java.util.Arrays;

import elki.clustering.kmeans.initialization.KMeansInitialization;
import elki.data.Clustering;
import elki.data.DoubleVector;
import elki.data.NumberVector;
import elki.data.model.KMeansModel;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableDataStore;
import elki.database.datastore.WritableDoubleDataStore;
import elki.database.ids.DBIDIter;
import elki.database.relation.Relation;
import elki.distance.UnitLengthEuclidianDistance;
import elki.logging.Logging;
import elki.math.DotProduct;
import elki.math.linearalgebra.VMath;
import elki.utilities.documentation.Reference;
import elki.utilities.optionhandling.parameterization.Parameterization;

import net.jafama.FastMath;

/**
 * Elkan's fast k-means by exploiting the triangle inequality.
 * <p>
 * This variant needs O(n*k) additional memory to store bounds.
 * <p>
 * See {@link HamerlyKMeans} for a close variant that only uses O(n*2)
 * additional memory for bounds.
 * <p>
 * Reference:
 * <p>
 * C. Elkan<br>
 * Using the triangle inequality to accelerate k-means<br>
 * Proc. 20th International Conference on Machine Learning, ICML 2003
 *
 * @author Erich Schubert
 * @since 0.7.0
 *
 * @navassoc - - - KMeansModel
 *
 * @param <V> vector datatype
 */
@Reference(authors = "C. Elkan", //
    title = "Using the triangle inequality to accelerate k-means", //
    booktitle = "Proc. 20th International Conference on Machine Learning, ICML 2003", //
    url = "http://www.aaai.org/Library/ICML/2003/icml03-022.php", //
    bibkey = "DBLP:conf/icml/Elkan03")
public class ElkanSphericalKMeans2<V extends NumberVector> extends AbstractKMeans<V, KMeansModel> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(ElkanSphericalKMeans2.class);

  /**
   * Flag whether to compute the final variance statistic.
   */
  protected boolean varstat = false;

  /**
   * Constructor.
   *
   * @param distance distance function
   * @param k k parameter
   * @param maxiter Maxiter parameter
   * @param initializer Initialization method
   * @param varstat Compute the variance statistic
   */
  public ElkanSphericalKMeans2(int k, int maxiter, KMeansInitialization initializer, boolean varstat) {
    super(UnitLengthEuclidianDistance.STATIC, k, maxiter, initializer);
    this.varstat = varstat;
  }

  @Override
  public Clustering<KMeansModel> run(Relation<V> relation) {
    Instance instance = new Instance(relation, initialMeans(relation));
    instance.run(maxiter);
    return instance.buildResult(varstat, relation);
  }

  /**
   * Inner instance, storing state for a single data set.
   *
   * @author Alexander Voﬂ
   */
  protected static class Instance extends AbstractKMeans.Instance {
    /**
     * Upper bounds
     */
    WritableDoubleDataStore upper;

    /**
     * Lower bounds
     */
    WritableDataStore<double[]> lower;

    /**
     * Sums of clusters.
     */
    double[][] sums;

    /**
     * Scratch space for new means.
     */
    double[][] newmeans;

    /**
     * Cluster separation
     */
    double[] sep = new double[k];

    /**
     * Cluster center distances
     */
    double[][] cdist = new double[k][k];

    /**
     * Cluster center similarities
     */
    double[][] csim = new double[k][k];

    /**
     * Constructor.
     *
     * @param relation Relation
     * @param means Initial means
     */
    public Instance(Relation<? extends NumberVector> relation, double[][] means) {
      super(relation, UnitLengthEuclidianDistance.STATIC, means);
      upper = DataStoreUtil.makeDoubleStorage(relation.getDBIDs(), DataStoreFactory.HINT_TEMP | DataStoreFactory.HINT_HOT, Double.POSITIVE_INFINITY);
      lower = DataStoreUtil.makeStorage(relation.getDBIDs(), DataStoreFactory.HINT_TEMP | DataStoreFactory.HINT_HOT, double[].class);
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        lower.put(it, new double[k]); // Filled with 0.
      }
      final int dim = means[0].length;
      sums = new double[k][dim];
      newmeans = new double[k][dim];
      sep = new double[k];
      cdist = new double[k][k];
      csim = new double[k][k];
    }

    @Override
    protected int iterate(int iteration) {
      if(iteration == 1) {
        return initialAssignToNearestCluster();
      }
      meansFromSums(newmeans, sums);
      movedDistance(means, newmeans, sep);
      updateBounds(sep);
      copyMeans(newmeans, means);
      return assignToNearestCluster();
    }

    /**
     * Compute means from cluster sums by averaging.
     * 
     * @param dst Output means
     * @param sums Input sums
     */
    protected void meansFromSums(double[][] dst, double[][] sums) {
      for(int i = 0; i < k; i++) {
        double length = .0;
        for(double d : sums[i]) {
          length += d * d;
        }
        length = FastMath.sqrt(length);
        VMath.overwriteTimes(dst[i], sums[i], 1. / length);
      }
    }

    protected int initialAssignToNearestCluster() {
      assert k == means.length;
      initialSeperation(cdist);
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        NumberVector fv = relation.get(it);
        double[] l = lower.get(it);
        // Check all (other) means:
        double best = l[0] = distance(fv, means[0]);
        int minIndex = 0;
        for(int j = 1; j < k; j++) {
          if(best > cdist[minIndex][j]) {
            double dist = l[j] = distance(fv, means[j]);
            if(dist < best) {
              minIndex = j;
              best = dist;
            }
          }
        }
        for(int j = 1; j < k; j++) {
          if(l[j] == 0. && j != minIndex) {
            l[j] = 2 * cdist[minIndex][j] - best;
          }
        }
        // Assign to nearest cluster.
        clusters.get(minIndex).add(it);
        assignment.putInt(it, minIndex);
        upper.putDouble(it, best);
        plusEquals(sums[minIndex], fv);
      }
      return relation.size();
    }

    @Override
    protected int assignToNearestCluster() {
      assert (k == means.length);
      recomputeSeparation(); // #1
      int changed = 0;
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        final int orig = assignment.intValue(it);
        double u = upper.doubleValue(it);
        // Upper bound check (#2):
        if(u <= sep[orig]) {
          continue;
        }
        boolean recomputeDistance = true; // Elkan's r(x)
        double curSim = 0.;
        NumberVector fv = relation.get(it);
        double[] l = lower.get(it);
        // Check all (other) means:
        int cur = orig;
        for(int j = 0; j < k; j++) {
          if(orig == j || u <= l[j] || u <= cdist[cur][j]) {
            continue; // Condition #3 i-iii not satisfied
          }
          if(recomputeDistance) {
            curSim = similarity(fv, means[cur]);
            u = distanceFromSimilarity(curSim);
            upper.putDouble(it, u);
            recomputeDistance = false; // Once only
          }
          if(curSim >= csim[cur][j] || u <= l[j]) {
            continue;
          }
          double sim = similarity(fv, means[j]);
          double dist = distanceFromSimilarity(sim);
          l[j] = dist;
          if(sim > curSim) {
            cur = j;
            curSim = sim;
            u = dist;
          }
        }
        // Object is to be reassigned.
        if(cur != orig) {
          upper.putDouble(it, u); // Remember bound.
          clusters.get(cur).add(it);
          clusters.get(orig).remove(it);
          assignment.putInt(it, cur);
          plusMinusEquals(sums[cur], sums[orig], fv);
          ++changed;
        }
      }
      // assert noAssignmentWrong();
      return changed;
    }

    private boolean noAssignmentWrong() {
      int changed = 0;
      for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
        NumberVector fv = relation.get(iditer);
        int cur = assignment.intValue(iditer);
        double curSim = similarity(fv, means[cur]);
        int maxIndex = cur;
        double maxSim = curSim;
        int maxIndex2 = 0;
        double maxSim2 = 1;
        for(int i = 0; i < k; i++) {
          if(i == cur) {
            continue;
          }
          double sim = similarity(fv, means[i]);
          if(sim > maxSim) {
            maxIndex2 = maxIndex;
            maxIndex = i;
            maxSim2 = maxSim;
            maxSim = sim;
          }
          else if(sim > maxSim2) {
            maxSim2 = sim;
            maxIndex2 = k;
          }
        }
        if(maxIndex != cur) {
          changed++;

          LOG.error(iditer.internalGetIndex() + " : ");
          LOG.error("curDist (" + cur + ") : " + distanceFromSimilarity(curSim));
          LOG.error("actDist (" + maxIndex + ") : " + distanceFromSimilarity(maxSim));
          LOG.error("secDist (" + maxIndex2 + ") : " + distanceFromSimilarity(maxSim2));
          double lowerVal = lower.get(iditer)[cur];
          double lowerTrue = distanceFromSimilarity(maxSim2);
          LOG.error("lower : " + lowerVal + " actually : " + lowerTrue + " isCorrect : " + (lowerVal <= lowerTrue));

          double upperVal = upper.doubleValue(iditer);
          double upperTrue = distanceFromSimilarity(maxSim);
          LOG.error("upper : " + upperVal + " actually : " + upperTrue + " isCorrect : " + (upperVal >= upperTrue));
          LOG.error("boundCheck : " + (upperVal <= lowerVal) + " actually : " + (upperTrue <= upperVal));
        }
      }
      LOG.error("wrong assignments : " + changed);
      // return changed == 0;
      return true;
    }

    protected double similarity(NumberVector vec1, double[] vec2) {
      diststat++;
      return DotProduct.dot(vec1, vec2);
    }

    protected double similarity(double[] vec1, double[] vec2) {
      return similarity(new DoubleVector(vec1), vec2);
    }

    /**
     * Update the bounds for k-means.
     *
     * @param move Movement of centers
     */
    protected void updateBounds(double[] move) {
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        upper.increment(it, move[assignment.intValue(it)]);
        VMath.minusEquals(lower.get(it), move);
      }
    }

    /**
     * Initial separation of means. Used by Elkan, SimplifiedElkan.
     *
     * @param cdist Pairwise separation output (as sqrt/2)
     */
    protected void initialSeperation(double[][] cdist) {
      final int k = means.length;
      for(int i = 1; i < k; i++) {
        double[] mi = means[i];
        for(int j = 0; j < i; j++) {
          csim[i][j] = csim[j][i] = similarity(mi, means[j]);
          cdist[i][j] = cdist[j][i] = .5 * distanceFromSimilarity(csim[i][j]);
        }
      }
    }

    protected void recomputeSeparation() {
      final int k = means.length;
      assert sep.length == k;
      Arrays.fill(sep, Double.POSITIVE_INFINITY);
      for(int i = 1; i < k; i++) {
        double[] mi = means[i];
        for(int j = 0; j < i; j++) {
          double sim = similarity(mi, means[j]);
          csim[i][j] = csim[j][i] = (similarity(mi, means[j]) + 3) * 1. / 4.;
          double halfd = 0.5 * distanceFromSimilarity(sim);
          cdist[i][j] = cdist[j][i] = halfd;
          sep[i] = (halfd < sep[i]) ? halfd : sep[i];
          sep[j] = (halfd < sep[j]) ? halfd : sep[j];
        }
      }
    }

    protected double distanceFromSimilarity(double sim) {
      return FastMath.sqrt(1 - sim);
    }

    @Override
    protected Logging getLogger() {
      return LOG;
    }
  }

  @Override
  protected Logging getLogger() {
    return LOG;
  }

  /**
   * Parameterization class.
   */
  public static class Par<V extends NumberVector> extends AbstractKMeans.Par<V> {
    @Override
    public void configure(Parameterization config) {
      super.configure(config);
      super.getParameterVarstat(config);
    }

    @Override
    public ElkanSphericalKMeans2<V> make() {
      return new ElkanSphericalKMeans2<>(k, maxiter, initializer, varstat);
    }
  }
}
