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
import elki.data.NumberVector;
import elki.data.model.KMeansModel;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableIntegerDataStore;
import elki.database.ids.DBIDIter;
import elki.database.relation.Relation;
import elki.logging.Logging;
import elki.utilities.datastructures.arrays.DoubleIntegerArrayQuickSort;
import elki.utilities.documentation.Reference;

import net.jafama.FastMath;

/**
 * Newlings's exponion k-means algorithm, exploiting the triangle inequality.
 * <p>
 * This is rewritten for a spherical k-Means.
 * <p>
 * Reference:
 * <p>
 * J. Newling<br>
 * Fast k-means with accurate bounds<br>
 * Proc. 33nd Int. Conf. on Machine Learning, ICML 2016
 *
 * @author Alexander Voﬂ
 *
 * @param <V> vector datatype
 */
@Reference(authors = "J. Newling", //
    title = "Fast k-means with accurate bounds", //
    booktitle = "Proc. 33nd Int. Conf. on Machine Learning, ICML 2016", //
    url = "http://jmlr.org/proceedings/papers/v48/newling16.html", //
    bibkey = "DBLP:conf/icml/NewlingF16")
public class ExponionSphericalKMeans<V extends NumberVector> extends HamerlySphericalKMeans<V> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(ExponionSphericalKMeans.class);

  /**
   * Constructor.
   *
   * @param k k parameter
   * @param maxiter Maxiter parameter
   * @param initializer Initialization method
   * @param varstat Compute the variance statistic
   */
  public ExponionSphericalKMeans(int k, int maxiter, KMeansInitialization initializer, boolean varstat) {
    super(k, maxiter, initializer, varstat);
  }

  @Override
  public Clustering<KMeansModel> run(Relation<V> relation) {
    Instance instance = new Instance(relation, initialMeans(relation));
    instance.run(maxiter);
    // instance.printAssignments();
    return instance.buildResult(varstat, relation);
  }

  /**
   * Inner instance, storing state for a single data set.
   */
  protected static class Instance extends HamerlySphericalKMeans.Instance {
    /**
     * Second nearest cluster.
     */
    WritableIntegerDataStore second;

    /**
     * Cluster center distances.
     */
    double[][] cdist;

    /**
     * Cluster center similarities
     */
    double[][] csim = new double[k][k];

    /**
     * partially sorted neighbors
     */
    int[][] cnum;

    public Instance(Relation<? extends NumberVector> relation, double[][] means) {
      super(relation, means);
      second = DataStoreUtil.makeIntegerStorage(relation.getDBIDs(), DataStoreFactory.HINT_TEMP | DataStoreFactory.HINT_HOT, -1);
      cdist = new double[k][k];
      cnum = new int[k][k - 1];
      for(int x = 0; x < k; x++) {
        for(int y = 0; y < k - 1; y++) {
          cnum[x][y] = y >= x ? y + 1 : y;
        }
      }
    }

    /**
     * Recompute the separation of cluster means.
     * <p>
     * Used by sort, and our exponion implementation.
     *
     * @param cdist Center-to-Center distances
     * @param cnum Center numbers
     */
    protected void sortMeans() {
      final int k = csim.length;
      double[] buf = new double[k - 1];
      for(int i = 0; i < k; i++) {
        System.arraycopy(csim[i], 0, buf, 0, i);
        System.arraycopy(csim[i], i + 1, buf, i, k - i - 1);
        for(int j = 0; j < buf.length; j++) {
          cnum[i][j] = j < i ? j : (j + 1);
        }
        DoubleIntegerArrayQuickSort.sortReverse(buf, cnum[i], k - 1);
      }
    }

    @Override
    protected int assignToNearestCluster() {
      assert (k == means.length);
      recomputeSeparation();
      sortMeans();
      int changed = 0;
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        final int cur = assignment.intValue(it);
        // Compute the current bound:
        final double lowerBound = lower.doubleValue(it);
        double upperBound = upper.doubleValue(it);
        if(upperBound <= lowerBound || upperBound <= sep[cur]) {
          continue;
        }
        // Update the upper bound
        NumberVector fv = relation.get(it);
        double curSim = similarity(fv, means[cur]);
        double curDist = distanceFromSimilarity(curSim);
        upperBound = curDist;
        upper.putDouble(it, upperBound);
        if(curSim >= sepSim[cur] || upperBound <= lowerBound) {
          continue;
        }
        // Find closest center, and distance to two closest centers
        double max1 = curSim, max2 = Double.NEGATIVE_INFINITY;
        int maxIndex = cur;
        double r = 2 * (upperBound + sep[cur]);

        for(int i = 0; i < k - 1; i++) {
          int c = cnum[cur][i];
          if(cdist[cur][c] > r) {
            break;
          }
          double sim = similarity(fv, means[c]);
          if(sim > max1) {
            maxIndex = c;
            max2 = max1;
            max1 = sim;
          }
          else if(sim > max2) {
            max2 = sim;
          }
        }
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
      // printAssignments();
      // assert noAssignmentWrong();
      return changed;
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
  public static class Par<V extends NumberVector> extends HamerlySphericalKMeans.Par<V> {

    @Override
    public ExponionSphericalKMeans<V> make() {
      return new ExponionSphericalKMeans<>(k, maxiter, initializer, varstat);
    }
  }
}
