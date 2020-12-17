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
 * This is <b>not</b> a complete implementation, the approximative sorting part
 * is missing. We also had to guess on the paper how to make best use of F.
 * <p>
 * Reference:
 * <p>
 * J. Newling<br>
 * Fast k-means with accurate bounds<br>
 * Proc. 33nd Int. Conf. on Machine Learning, ICML 2016
 *
 * @author Erich Schubert
 * @since 0.7.5
 *
 * @navassoc - - - KMeansModel
 *
 * @param <V> vector datatype
 */
@Reference(authors = "J. Newling", //
    title = "Fast k-means with accurate bounds", //
    booktitle = "Proc. 33nd Int. Conf. on Machine Learning, ICML 2016", //
    url = "http://jmlr.org/proceedings/papers/v48/newling16.html", //
    bibkey = "DBLP:conf/icml/NewlingF16")
public class ExponionNaiveSphericalKMeans<V extends NumberVector> extends HamerlySphericalKMeans<V> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(ExponionNaiveSphericalKMeans.class);

  /**
   * Constructor.
   *
   * @param k k parameter
   * @param maxiter Maxiter parameter
   * @param initializer Initialization method
   * @param varstat Compute the variance statistic
   */
  public ExponionNaiveSphericalKMeans(int k, int maxiter, KMeansInitialization initializer, boolean varstat) {
    super(k, maxiter, initializer, varstat);
  }

  @Override
  public Clustering<KMeansModel> run(Relation<V> relation) {
    Instance instance = new Instance(relation, initialMeans(relation));
    instance.run(maxiter);
    instance.printAssignments();
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
      cnum = new int[k][k - 1];
      for(int x = 0; x < k; x++) {
        for(int y = 0; y < k - 1; y++) {
          cnum[x][y] = y >= x ? y + 1 : y;
        }
      }
    }

    @Override
    protected int assignToNearestCluster() {
      assert (k == means.length);
      recomputeSeparation();
      sort();
      int changed = 0;
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        final int cur = assignment.intValue(it);
        // Compute the current bound:
        final double lowerBound = lower.doubleValue(it);
        final double sa = sep[cur];
        double upperBound = upper.doubleValue(it);
        if(upperBound <= lowerBound) {
          continue;
        }
        if(upperBound <= sa) {
          continue;
        }
        // Update the upper bound
        NumberVector fv = relation.get(it);
        double curs2 = similarity(fv, means[cur]);
        double curd2 = distanceFromSimilarity(curs2);
        upperBound = curd2;
        upper.putDouble(it, upperBound);
        if(upperBound <= lowerBound) {
          continue;
        }
        if(upperBound <= sa) {
          continue;
        }
        // Find closest center, and distance to two closest centers
        double max1 = curs2, max2 = Double.NEGATIVE_INFINITY;
        int maxIndex = cur;
        double r = 1 - FastMath.pow2(2 * (upperBound + sa));
        // System.out.println("maxJ : " + maxJ);
        for(int i = 0; i < k - 1; i++) {
          int c = cnum[cur][i];
          if(csim[cur][c] < r) {
            int pruned = k - 1 - i;
            System.out.println("pruned " + pruned);
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
          upper.putDouble(it, max1 == curs2 ? upperBound : distanceFromSimilarity(max1));
        }
        lower.putDouble(it, max2 == curs2 ? upperBound : distanceFromSimilarity(max2));
      }
      // printAssignments();
      // assert noAssignmentWrong();
      return changed;
    }

    protected void sort() {
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

    /**
     * Sorts nums so that all indices j' in nums with values[j'] == 0 are at the
     * end of nums. Returns the index of the first j'. Returns -1 if values does
     * not contain a 1.
     */
    protected int presort(int[] nums, double[] values) {
      int left = 0, right = nums.length - 1;
      while(left < right) {
        // move left to first 1
        while(values[nums[left]] != 0. && left < right) {
          left++;
        }
        // move right to next non 1
        while(values[nums[right]] == 0. && right > left) {
          right--;
        }
        swap(nums, left, right);

        // move left and right
        right--;
        left++;
      }
      // array contains no 1
      if(left >= nums.length || right <= 0) {
        return -1;
      }
      if(values[nums[right]] == 0.) {
        return right;
      }
      return right + 1;
    }

    /**
     * Swaps the elements arr[ind1] and arr[ind2];
     */
    private void swap(int[] arr, int ind1, int ind2) {
      int val1 = arr[ind1];
      arr[ind1] = arr[ind2];
      arr[ind2] = val1;
    }

    protected void recomputeSeparation() {
      final int k = means.length;
      assert sep.length == k;
      Arrays.fill(sep, Double.NEGATIVE_INFINITY);
      for(int i = 1; i < k; i++) {
        double[] mi = means[i];
        for(int j = 0; j < i; j++) {
          double sim = similarity(mi, means[j]);
          csim[i][j] = csim[j][i] = sim;
          sep[i] = (sim > sep[i]) ? sim : sep[i];
          sep[j] = (sim > sep[j]) ? sim : sep[j];
        }
      }
      for(int i = 0; i < k; i++) {
        sep[i] = 0.5 * distanceFromSimilarity(sep[i]);
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
   * @author Erich Schubert
   */
  public static class Par<V extends NumberVector> extends HamerlySphericalKMeans.Par<V> {

    @Override
    public ExponionNaiveSphericalKMeans<V> make() {
      return new ExponionNaiveSphericalKMeans<>(k, maxiter, initializer, varstat);
    }
  }
}
