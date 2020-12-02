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
import java.util.stream.Collectors;

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

    /**
     * outer radius of layer
     */
    double[][] eRadius;

    /**
     * first index in cnum where similarity is 0
     */
    int[] eMax;

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
      eRadius = new double[k][(int) Math.ceil((Math.log(k) / Math.log(2)))];
      eMax = new int[k];
    }

    @Override
    protected int assignToNearestCluster() {
      assert (k == means.length);
      recomputeSeparation();
      partialSort();
      nearestMeans(cdist, cnum);
      int changed = 0;
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        final int cur = assignment.intValue(it);
        // Compute the current bound:
        final double z = lower.doubleValue(it);
        final double sa = sep[cur];
        double u = upper.doubleValue(it);
        if(u <= z || u <= sa) {
           continue;
        }
        // Update the upper bound
        NumberVector fv = relation.get(it);
        double curs2 = similarity(fv, means[cur]);
        double curd2 = distanceFromSimilarity(curs2);
        u = curd2;
        upper.putDouble(it, u);
        if(u <= z || u <= sa) {
           continue;
        }
        // Find closest center, and distance to two closest centers
        double max1 = curs2, max2 = Double.NEGATIVE_INFINITY;
        int maxIndex = cur;
        double r = 2 * (u + sa); // Our cdist are scaled 0.5
        int maxJ = findJ(it, r);
        for(int i = 0; i < maxJ; i++) {
          int c = cnum[cur][i];
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
          upper.putDouble(it, max1 == curs2 ? u : distanceFromSimilarity(max1));
        }
        lower.putDouble(it, max2 == curs2 ? u : distanceFromSimilarity(max2));
      }
      return changed;
    }

    /**
     * Finds the index of the first
     * 
     * @param it
     * @param r
     * @return
     */
    private int findJ(DBIDIter it, double r) {
      if(r >= 1) {
        return k - 1;
      }
      int cur = assignment.intValue(it);
      double[] curE = eRadius[cur];
      int ind = 0;
      for(int f = 0; f < curE.length; f++) {
        if(r <= curE[ind]) {
          break;
        }
        ind++;
      }
      return Math.min((int) FastMath.twoPow(ind + 1) - 2, k - 1);
    }

    protected void partialSort() {
      System.err.println("********");
      System.err.println("Partial Sort");
      for(int j = 0; j < k; j++) {
        System.out.println(j + " : ");
        int right = cnum[j].length - 1;
        int annulusInd = eRadius[j].length - 1;
        int amount = (int) FastMath.twoPow(annulusInd + 1) - 2;
        System.out.println("ind : " + annulusInd);
        selectBiggest(cnum[j], csim[j], right, amount);
        double max = maxExceptJ(cnum[j], csim[j], j, amount, right);
        eRadius[j][annulusInd--] = distanceFromSimilarity(max);
        amount = (amount - 2) / 2;
        right = (int) FastMath.twoPow(annulusInd + 1) + 1;

        while(annulusInd-- >= 1) {
          System.out.println("ind : " + annulusInd);
          selectBiggest(cnum[j], csim[j], right, amount);
          max = maxExceptJ(cnum[j], csim[j], j, amount, right);
          eRadius[j][annulusInd] = distanceFromSimilarity(max);
          amount = (amount - 2) / 2;
          right = (int) FastMath.twoPow(annulusInd + 1) + 1;
        }

        max = maxExceptJ(cnum[j], csim[j], j, 0, 1);
        eRadius[j][0] = distanceFromSimilarity(max);
        System.out.println("e : " + Arrays.toString(eRadius[j]));
        final int jFinal = j;
        System.out.println("md: " + Arrays.stream(cnum[j]).boxed().map(jHat -> 2 * cdist[jFinal][jHat]).collect(Collectors.toList()).toString());
      }
      System.err.println("********");
    }

    private double maxExceptJ(int[] indices, double[] values, int j, int left, int right) {
      double max = values[indices[left]];
      for(int i = left + 1; i <= right; i++) {
        int index = indices[i];
        if(index == j) {
          continue;
        }
        if(values[index] > max) {
          max = values[index];
        }
      }
      return max;
    }

    private void selectBiggest(int[] indices, double values[], int right, int amount) {
      System.out.println("select " + amount + " from " + (right + 1));
      int left = 0;
      int goalIndex = amount - 1;
      while(true) {
        if(left >= right) {
          return;
        }

        int pivotIndex = left + (int) (Math.random() * (right - left));
        double pivot = values[indices[pivotIndex]];
        pivotIndex = partition(indices, values, left, right, pivot);

        if(pivotIndex == goalIndex) {
          return;
        }

        if(pivotIndex < goalIndex) {
          right = pivotIndex - 1;
        }
        else {
          left = pivotIndex + 1;
        }
      }
    }

    private int partition(int[] indices, double[] values, int left, int right, double pivot) {
      double minVal = Math.min(values[indices[left]], values[indices[right]]);
      while(true) {
        while(values[indices[left]] > pivot) {
          minVal = Math.min(minVal, values[left]);
          left++;
        }
        while(values[indices[right]] < pivot) {
          minVal = Math.min(minVal, values[right]);
          right--;
        }
        if(left >= right) {
          break;
        }
        swap(indices, left, right);
      }
      return right;
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
      Arrays.fill(sep, Double.POSITIVE_INFINITY);
      for(int i = 1; i < k; i++) {
        double[] mi = means[i];
        for(int j = 0; j < i; j++) {
          double sim = similarity(mi, means[j]);
          csim[i][j] = csim[j][i] = sim;
          double halfd = 0.5 * distanceFromSimilarity(sim);
          cdist[i][j] = cdist[j][i] = halfd;
          sep[i] = sep[j] = (halfd < sep[i]) ? halfd : sep[i];
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
   *
   * @author Erich Schubert
   */
  public static class Par<V extends NumberVector> extends HamerlySphericalKMeans.Par<V> {
    @Override
    public ExponionSphericalKMeans<V> make() {
      return new ExponionSphericalKMeans<>(k, maxiter, initializer, varstat);
    }
  }
}
