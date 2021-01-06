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
import elki.data.SparseDoubleVector;
import elki.data.model.KMeansModel;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableDoubleDataStore;
import elki.database.ids.DBIDIter;
import elki.database.relation.Relation;
import elki.distance.UnitLengthEuclidianDistance;
import elki.logging.Logging;
import elki.math.DotProduct;
import elki.math.linearalgebra.VMath;
import elki.utilities.optionhandling.parameterization.Parameterization;

import net.jafama.FastMath;

public class HamerlySphericalKMeans2<V extends NumberVector> extends AbstractKMeans<V, KMeansModel> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(HamerlySphericalKMeans2.class);

  /**
   * Flag whether to compute the final variance statistic.
   */
  protected boolean varstat = true;

  /**
   * Constructor.
   *
   * @param distance distance function
   * @param k k parameter
   * @param maxiter Maxiter parameter
   * @param initializer Initialization method
   * @param varstat Compute the variance statistic
   */
  public HamerlySphericalKMeans2(int k, int maxiter, KMeansInitialization initializer, boolean varstat) {
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
   */
  protected static class Instance extends AbstractKMeans.Instance {

    /**
     * Sum aggregate for the new mean.
     */
    double[][] sums;

    /**
     * Temporary storage for the new means.
     */
    double[][] newmeans;

    /**
     * Separation of means / distance moved.
     */
    double[] sepDist;

    /**
     * Separation of means / distance moved.
     */
    double[] sepSim;

    double[] movedDistances;

    /**
     * Upper bounding distance
     */
    WritableDoubleDataStore upper;

    /**
     * Lower bounding distance
     */
    WritableDoubleDataStore lower;

    /**
     * Constructor.
     *
     * @param relation Relation
     * @param means Initial means
     */
    public Instance(Relation<? extends NumberVector> relation, double[][] means) {
      super(relation, UnitLengthEuclidianDistance.STATIC, means);
      upper = DataStoreUtil.makeDoubleStorage(relation.getDBIDs(), DataStoreFactory.HINT_TEMP | DataStoreFactory.HINT_HOT, Double.POSITIVE_INFINITY);
      lower = DataStoreUtil.makeDoubleStorage(relation.getDBIDs(), DataStoreFactory.HINT_TEMP | DataStoreFactory.HINT_HOT, 0.);
      final int dim = means[0].length;
      sums = new double[k][dim];
      newmeans = new double[k][dim];
      sepDist = new double[k];
      sepSim = new double[k];
      movedDistances = new double[k];
      // assert normalized();
    }

    private boolean normalized() {
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        NumberVector fv = relation.get(it);
        if(!normalized(fv))
          return false;
      }
      return true;
    }

    private boolean normalized(NumberVector fv) {
      double len = 0;
      for(double d : fv.toArray()) {
        len += d * d;
      }
      len = FastMath.sqrt(len);
      return len <= 1.0 + 3.0 * Double.MIN_VALUE || len >= 1.0 - 3.0 * Double.MIN_VALUE;
    }

    @Override
    protected int iterate(int iteration) {
      if(iteration == 1) {
        return initialAssignToNearestCluster();
      }
      meansFromSums(newmeans, sums);
      updateBounds(sepDist, movedDistance(means, newmeans, sepDist));
      copyMeans(newmeans, means);
      return assignToNearestCluster();
    }

    /**
     * Maximum distance moved.
     * <p>
     * Used by Hamerly, Elkan (not using the maximum).
     *
     * @param means Old means
     * @param newmeans New means
     * @param dists Distances moved (output)
     * @return Maximum distance moved
     */
    protected double movedDistance(double[][] means, double[][] newmeans, double[] dists) {
      assert newmeans.length == means.length && dists.length == means.length;
      double max = 0.;
      for(int i = 0; i < means.length; i++) {
        double d = dists[i] = movedDistances[i] = distance(new SparseDoubleVector(means[i]), newmeans[i]);
        max = (d > max) ? d : max;
      }
      return max;
    }

    protected double similarity(NumberVector vec1, double[] vec2) {
      diststat++;
      return DotProduct.dot(vec1, vec2);
    }

    protected double similarity(double[] vec1, double[] vec2) {
      return similarity(new DoubleVector(vec1), vec2);
    }

    protected double distanceFromSimilarity(double sim) {
      return FastMath.sqrt(1 - sim);
    }

    protected double similarityFromDistance(double dist) {
      return 1 - dist * dist;
    }

    protected void computeSeparationSimilarity(double[][] cost) {
      for(int i = 0; i < k; i++) {
        double[] mi = means[i];
        for(int j = 0; j < i; j++) {
          cost[i][j] = cost[j][i] = similarity(mi, means[j]);
        }
      }
    }

    /**
     * Perform initial cluster assignment.
     *
     * @return Number of changes (i.e. relation size)
     */
    // TODO add pruning correctly
    protected int initialAssignToNearestCluster() {
      assert k == means.length;
      double[][] cdist = new double[k][k];
      computeSeparationSimilarity(cdist);
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        NumberVector fv = relation.get(it);
        // Find closest center, and distance to two closest centers
        double max1 = similarity(fv, means[0]), max2 = similarity(fv, means[1]);
        int maxIndex = 0;
        if(max2 > max1) {
          double tmp = max1;
          max1 = max2;
          max2 = tmp;
          maxIndex = 1;
        }
        for(int i = 2; i < k; i++) {
          // if(max2 < cdist[maxIndex][i]) {
          double sim = similarity(fv, means[i]);
          if(sim > max1) {
            maxIndex = i;
            max2 = max1;
            max1 = sim;
          }
          else if(sim > max2) {
            max2 = sim;
          }
          // }
        }
        // Assign to nearest cluster.
        clusters.get(maxIndex).add(it);
        assignment.putInt(it, maxIndex);
        plusEquals(sums[maxIndex], fv);
        upper.putDouble(it, distanceFromSimilarity(max1));
        lower.putDouble(it, distanceFromSimilarity(max2));
      }
      // printAssignments();
      return relation.size();
    }

    @Override
    protected int assignToNearestCluster() {
      assert (k == means.length);
      recomputeSeparation(means, sepDist, sepSim);
      int changed = 0;
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        final int cur = assignment.intValue(it);
        // Compute the current bound:
        final double saDist = sepDist[cur];
        final double saSim = sepSim[cur];
        double lowerBound = lower.doubleValue(it);
        double upperBound = upper.doubleValue(it);
        if(upperBound <= lowerBound || upperBound <= saDist) {
          continue;
        }
        // Update the upper bound
        NumberVector fv = relation.get(it);
        upperBound = similarity(fv, means[cur]);
        if(upperBound >= saSim || upperBound >= similarityFromDistance(lowerBound)) {
          upper.putDouble(it, distanceFromSimilarity(upperBound));
          continue;
        }
        // Find closest center, and distance to two closest centers
        double curSim = upperBound, max1 = upperBound,
            max2 = Double.NEGATIVE_INFINITY;
        int maxIndex = cur;
        for(int i = 0; i < k; i++) {
          if(i == cur) {
            continue;
          }
          double sim = similarity(fv, means[i]);
          if(sim > max1) {
            maxIndex = i;
            max2 = max1;
            max1 = sim;
          }
          else if(sim > max2) {
            max2 = sim;
          }
        }
        upperBound = distanceFromSimilarity(upperBound);
        if(maxIndex != cur) {
          clusters.get(maxIndex).add(it);
          clusters.get(cur).remove(it);
          assignment.putInt(it, maxIndex);
          plusMinusEquals(sums[maxIndex], sums[cur], fv);
          ++changed;
          upper.putDouble(it, max1 == curSim ? upperBound : distanceFromSimilarity(max1));
        }
        lower.putDouble(it, max2 == curSim ? upperBound : distanceFromSimilarity(max2));
      }
      // assert noAssignmentWrong();
      // printAssignments();
      return changed;
    }

    protected boolean noAssignmentWrong() {
      int changed = 0;
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        NumberVector fv = relation.get(it);
        int cur = assignment.intValue(it);
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

          LOG.error(it.internalGetIndex() + " : ");
          LOG.error("curDist (" + cur + ") : " + distanceFromSimilarity(curSim));
          LOG.error("movedDist (" + cur + ") : " + movedDistances[cur]);
          LOG.error("actDist (" + maxIndex + ") : " + distanceFromSimilarity(maxSim));
          LOG.error("movedDist (" + maxIndex + ") : " + movedDistances[maxIndex]);
          LOG.error("secDist (" + maxIndex2 + ") : " + distanceFromSimilarity(maxSim2));
          double lowerVal = lower.doubleValue(it);
          double lowerTrue = distanceFromSimilarity(maxSim2);
          LOG.error("lower : " + lowerVal + " actually : " + lowerTrue + " isCorrect : " + (lowerVal <= lowerTrue));

          double upperVal = upper.doubleValue(it);
          double upperTrue = distanceFromSimilarity(maxSim);
          LOG.error("upper : " + upperVal + " actually : " + upperTrue + " isCorrect : " + (upperVal >= upperTrue));
          LOG.error("boundCheck : " + (upperVal <= lowerVal) + " actually : " + (upperTrue <= lowerTrue));
        }
      }
      LOG.error("wrong assignments : " + changed);
      return changed == 0;
      // return true;
    }

    /**
     * Recompute the separation of cluster means.
     *
     * @param means Means
     * @param sepDist Output array of separation (half-sqrt scaled)
     */
    protected void recomputeSeparation(double[][] means, double[] sepDist, double[] sepSim) {
      final int k = means.length;
      assert sepDist.length == k;
      Arrays.fill(sepDist, Double.NEGATIVE_INFINITY);
      // First find max Similarity
      for(int i = 1; i < k; i++) {
        double[] m1 = means[i];
        for(int j = 0; j < i; j++) {
          double curSim = similarity(m1, means[j]);
          sepSim[i] = (curSim > sepSim[i]) ? curSim : sepSim[i];
          sepSim[j] = (curSim > sepSim[j]) ? curSim : sepSim[j];
        }
      }
      // Now translate to 1-(.5*sqrt(1-a))^2
      for(int i = 0; i < k; i++) {
        sepDist[i] = .5 * distanceFromSimilarity(sepSim[i]);
        sepSim[i] = (sepSim[i] + 3) * .25;
      }
    }

    /**
     * Compute means from cluster sums by adding and normalizing.
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
        if(length == 0) {
          System.arraycopy(sums[i], 0, dst[i], 0, sums[i].length);
        }
        else {
          VMath.overwriteTimes(dst[i], sums[i], 1. / length);
        }
      }
    }

    /**
     * Update the bounds for k-means.
     *
     * @param move Movement of centers
     * @param delta Maximum center movement.
     */
    protected void updateBounds(double[] move, double delta) {
      delta = -delta;
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        upper.increment(it, move[assignment.intValue(it)]);
        lower.increment(it, delta);
        if(lower.doubleValue(it) < 0.) {
          lower.putDouble(it, 0.);
        }
      }
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
    protected boolean needsMetric() {
      return true;
    }

    @Override
    public void configure(Parameterization config) {
      super.configure(config);
      super.getParameterVarstat(config);
    }

    @Override
    public HamerlySphericalKMeans2<V> make() {
      return new HamerlySphericalKMeans2<>(k, maxiter, initializer, varstat);
    }
  }
}
