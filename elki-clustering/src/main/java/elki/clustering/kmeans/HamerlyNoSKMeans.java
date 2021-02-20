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

import elki.clustering.kmeans.initialization.KMeansInitialization;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.KMeansModel;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableDoubleDataStore;
import elki.database.ids.DBIDIter;
import elki.database.relation.Relation;
import elki.distance.NumberVectorDistance;
import elki.logging.Logging;
import elki.utilities.documentation.Reference;
import elki.utilities.optionhandling.parameterization.Parameterization;

import net.jafama.FastMath;

/**
 * Hamerly's fast k-means by exploiting the triangle inequality. This version
 * only relies on bounds and does not use inter centroid distances.
 * <p>
 * Reference:
 * <p>
 * G. Hamerly<br>
 * Making k-means even faster<br>
 * Proc. 2010 SIAM International Conference on Data Mining
 *
 * @author Alexander Voﬂ
 * @since 0.7.0
 *
 * @navassoc - - - KMeansModel
 *
 * @param <V> vector datatype
 */
@Reference(authors = "G. Hamerly", //
    title = "Making k-means even faster", //
    booktitle = "Proc. 2010 SIAM International Conference on Data Mining", //
    url = "https://doi.org/10.1137/1.9781611972801.12", //
    bibkey = "DBLP:conf/sdm/Hamerly10")
public class HamerlyNoSKMeans<V extends NumberVector> extends AbstractKMeans<V, KMeansModel> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(HamerlyNoSKMeans.class);

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
  public HamerlyNoSKMeans(NumberVectorDistance<? super V> distance, int k, int maxiter, KMeansInitialization initializer, boolean varstat) {
    super(distance, k, maxiter, initializer);
    this.varstat = varstat;
  }

  @Override
  public Clustering<KMeansModel> run(Relation<V> relation) {
    Instance instance = new Instance(relation, distance, initialMeans(relation));
    instance.run(maxiter);
    return instance.buildResult(varstat, relation);
  }

  /**
   * Inner instance, storing state for a single data set.
   *
   * @author Erich Schubert
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
     * distance moved.
     */
    double[] sep;

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
    public Instance(Relation<? extends NumberVector> relation, NumberVectorDistance<?> df, double[][] means) {
      super(relation, df, means);
      upper = DataStoreUtil.makeDoubleStorage(relation.getDBIDs(), DataStoreFactory.HINT_TEMP | DataStoreFactory.HINT_HOT, Double.POSITIVE_INFINITY);
      lower = DataStoreUtil.makeDoubleStorage(relation.getDBIDs(), DataStoreFactory.HINT_TEMP | DataStoreFactory.HINT_HOT, 0.);
      final int dim = means[0].length;
      sums = new double[k][dim];
      newmeans = new double[k][dim];
      sep = new double[k];
    }

    @Override
    protected int iterate(int iteration) {
      if(iteration == 1) {
        return initialAssignToNearestCluster();
      }
      meansFromSums(newmeans, sums);
      updateBounds(sep, movedDistance(means, newmeans, sep));
      copyMeans(newmeans, means);
      return assignToNearestCluster();
    }

    /**
     * Perform initial cluster assignment.
     *
     * @return Number of changes (i.e. relation size)
     */
    protected int initialAssignToNearestCluster() {
      assert k == means.length;
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        NumberVector fv = relation.get(it);
        // Find closest center, and distance to two closest centers
        double min1 = distance(fv, means[0]), min2 = distance(fv, means[1]);
        int minIndex = 0;
        if(min2 < min1) {
          double tmp = min1;
          min1 = min2;
          min2 = tmp;
          minIndex = 1;
        }
        for(int i = 2; i < k; i++) {
          double dist = distance(fv, means[i]);
          if(dist < min1) {
            minIndex = i;
            min2 = min1;
            min1 = dist;
          }
          else if(dist < min2) {
            min2 = dist;
          }
        }
        // Assign to nearest cluster.
        clusters.get(minIndex).add(it);
        assignment.putInt(it, minIndex);
        plusEquals(sums[minIndex], fv);
        upper.putDouble(it, isSquared ? FastMath.sqrt(min1) : min1);
        lower.putDouble(it, isSquared ? FastMath.sqrt(min2) : min2);
      }
      return relation.size();
    }

    @Override
    protected int assignToNearestCluster() {
      assert (k == means.length);
      int changed = 0;
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        final int cur = assignment.intValue(it);
        // Compute the current bound:
        final double z = lower.doubleValue(it);
        double u = upper.doubleValue(it);
        if(u <= z) {
          continue;
        }
        // Update the upper bound
        NumberVector fv = relation.get(it);
        double curd2 = distance(fv, means[cur]);
        u = isSquared ? FastMath.sqrt(curd2) : curd2;
        upper.putDouble(it, u);
        if(u <= z) {
          continue;
        }
        // Find closest center, and distance to two closest centers
        double min1 = curd2, min2 = Double.POSITIVE_INFINITY;
        int minIndex = cur;
        for(int i = 0; i < k; i++) {
          if(i == cur) {
            continue;
          }
          double dist = distance(fv, means[i]);
          if(dist < min1) {
            minIndex = i;
            min2 = min1;
            min1 = dist;
          }
          else if(dist < min2) {
            min2 = dist;
          }
        }
        if(minIndex != cur) {
          clusters.get(minIndex).add(it);
          clusters.get(cur).remove(it);
          assignment.putInt(it, minIndex);
          plusMinusEquals(sums[minIndex], sums[cur], fv);
          ++changed;
          upper.putDouble(it, min1 == curd2 ? u : isSquared ? FastMath.sqrt(min1) : min1);
        }
        lower.putDouble(it, min2 == curd2 ? u : isSquared ? FastMath.sqrt(min2) : min2);
      }
      return changed;
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
   *
   * @author Alexander Voﬂ
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
    public HamerlyNoSKMeans<V> make() {
      return new HamerlyNoSKMeans<>(distance, k, maxiter, initializer, varstat);
    }
  }
}
