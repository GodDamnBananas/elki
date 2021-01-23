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
import java.util.HashSet;
import java.util.Set;

import elki.clustering.kmeans.initialization.KMeansInitialization;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.KMeansModel;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableIntegerDataStore;
import elki.database.ids.DBIDIter;
import elki.database.relation.Relation;
import elki.distance.NumberVectorDistance;
import elki.logging.Logging;
import elki.utilities.documentation.Reference;

import net.jafama.FastMath;

/**
 * Newlings's exponion k-means algorithm, exploiting the triangle inequality.
 * This is not a complete implementation, the binary search was omitted for
 * performance reasons. With a realistic choice of k, the linear search is
 * faster.
 * <p>
 * Reference:
 * <p>
 * J. Newling<br>
 * Fast k-means with accurate bounds<br>
 * Proc. 33nd Int. Conf. on Machine Learning, ICML 2016
 *
 * @author Erich Schubert, Alexander Voﬂ
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
public class ExponionPartialSortKMeans<V extends NumberVector> extends HamerlyKMeans<V> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(ExponionPartialSortKMeans.class);

  /**
   * Constructor.
   *
   * @param distance distance function
   * @param k k parameter
   * @param maxiter Maxiter parameter
   * @param initializer Initialization method
   * @param varstat Compute the variance statistic
   */
  public ExponionPartialSortKMeans(NumberVectorDistance<? super V> distance, int k, int maxiter, KMeansInitialization initializer, boolean varstat) {
    super(distance, k, maxiter, initializer, varstat);
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
   * @author Erich Schubert, Alexander Voﬂ
   */
  protected static class Instance extends HamerlyKMeans.Instance {
    /**
     * Second nearest cluster.
     */
    WritableIntegerDataStore second;

    /**
     * Cluster center distances.
     */
    double[][] cdist;

    /**
     * partially sorted neighbors
     */
    int[][] cnum;

    /**
     * Outer radius of layers
     */
    double[][] eRadius;

    int[] radIndexToJ;

    public Instance(Relation<? extends NumberVector> relation, NumberVectorDistance<?> df, double[][] means) {
      super(relation, df, means);
      second = DataStoreUtil.makeIntegerStorage(relation.getDBIDs(), DataStoreFactory.HINT_TEMP | DataStoreFactory.HINT_HOT, -1);
      cdist = new double[k][k];
      cnum = new int[k][k - 1];
      for(int x = 0; x < k; x++) {
        for(int y = 0; y < k - 1; y++) {
          cnum[x][y] = y >= x ? y + 1 : y;
        }
      }
      int radLen = k > 6 ? (int) Math.ceil((Math.log(k - 2) / Math.log(2)) - 1) : 1;
      eRadius = new double[k][radLen];
      radIndexToJ = new int[radLen];
      int annSize = 2;
      int totalSize = annSize;
      for(int i = 0; i < radLen; i++) {
        radIndexToJ[i] = totalSize - 1;
        annSize *= 2;
        totalSize += annSize;
      }
    }

    @Override
    protected int initialAssignToNearestCluster() {
      assert k == means.length;
      computeSeparation(cdist);
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
          if(min2 > cdist[minIndex][i]) {
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
      recomputeSeparation(sep, cdist);
      partialSort();
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
        double curd2 = distance(fv, means[cur]);
        u = isSquared ? FastMath.sqrt(curd2) : curd2;
        upper.putDouble(it, u);
        if(u <= z || u <= sa) {
          continue;
        }
        double r = 2 * (u + sa); // Our cdist are scaled 0.5
        // Find closest center, and distance to two closest centers
        double min1 = curd2, min2 = Double.POSITIVE_INFINITY;
        int minIndex = cur;
        int maxJ = findJ(it, r);
        for(int i = 0; i < maxJ; i++) {
          int c = cnum[cur][i];
          double dist = distance(fv, means[c]);
          if(dist < min1) {
            minIndex = c;
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
      // assert noAssignmentWrong();
      return changed;
    }

    protected boolean noAssignmentWrong() {
      int changed = 0;
      for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
        NumberVector fv = relation.get(it);
        int cur = assignment.intValue(it);
        double curDist = distance(fv, means[cur]);
        int minIndex = cur;
        double maxDist = curDist;
        int minIndex2 = 0;
        double maxDist2 = Double.POSITIVE_INFINITY;
        for(int i = 0; i < k; i++) {
          if(i == cur) {
            continue;
          }
          double dist = distance(fv, means[i]);
          if(dist < maxDist) {
            minIndex2 = minIndex;
            minIndex = i;
            maxDist2 = maxDist;
            maxDist = dist;
          }
          else if(dist < maxDist2) {
            maxDist2 = dist;
            minIndex2 = i;
          }
        }
        if(minIndex != cur) {
          changed++;
        }
      }
      LOG.error("wrong assignments : " + changed);
      return changed == 0;
      // return true;
    }

    /**
     * Finds the index of the last element in the first layer that is bigger
     * than the given radius.
     * 
     * @param it
     * @param r
     * @return
     */
    // private int findJ(DBIDIter it, double r) {
    // int cur = assignment.intValue(it);
    // double[] curE = eRadius[cur];
    // if(r >= curE[curE.length - 1]) {
    // return k - 1;
    // }
    // if(r <= curE[0]) {
    // return 0;
    // }
    // // Perform a binary search with index m,
    // // such that curE[m] >= r && curE[m-1] < r
    // int left = 0;
    // int right = curE.length - 1;
    // int m = 0;
    // while(left <= right) {
    // m = (left + right) / 2;
    // if(curE[m] < r) {
    // left = m + 1;
    // }
    // else if(curE[m] >= r) {
    // if(curE[m - 1] < r) {
    // break;
    // }
    // else {
    // right = m - 1;
    // }
    // }
    // else {
    // break;
    // }
    // }
    // return Math.min((int) FastMath.twoPow(m + 1) + 2, k - 1);
    // }
    private int findJ(DBIDIter it, double r) {
      int cur = assignment.intValue(it);
      double[] curE = eRadius[cur];
      if(curE[curE.length - 1] <= r) {
        return k - 1;
      }
      int ind = 0;
      while(ind < curE.length - 1) {
        if(curE[ind] >= r) {
          break;
        }
        ind++;
      }
      return Math.min(radIndexToJ[ind], k - 1);

    }

    protected void partialSort() {
      for(int j = 0; j < k; j++) {
        // We have two annuli only when k > 3, so we can handle this case
        // directly
        if(k <= 3) {
          eRadius[j][0] = Arrays.stream(cdist[j]).max().getAsDouble();
          return;
        }
        int right = cnum[j].length - 1;
        int annulusInd = eRadius[j].length - 1;
        int amount = annulusInd > 1 ? radIndexToJ[annulusInd - 1] + 1 : 2;
        selectSmallest(cnum[j], cdist[j], right, amount);
        eRadius[j][annulusInd] = maxExceptJ(cnum[j], cdist[j], j, amount, right);

        while(annulusInd-- > 1) {
          right = amount - 1;
          amount = radIndexToJ[annulusInd - 1] + 1;
          selectSmallest(cnum[j], cdist[j], right, amount);
          eRadius[j][annulusInd] = maxExceptJ(cnum[j], cdist[j], j, amount, right);
        }
        eRadius[j][0] = maxExceptJ(cnum[j], cdist[j], j, 0, 1);
        // assert eRadiusCorrect(j);
      }
    }

    private boolean eRadiusCorrect(int curJ) {
      int[] curC = cnum[curJ];
      System.out.println(curJ + " : ");
      int annInd = 0;
      double eRadiusMax = 0;
      Set<Double> annSet = new HashSet<>();
      for(int j = 0; j < curC.length; j++) {
        final int hua = j;
        if(Arrays.stream(radIndexToJ).anyMatch(x -> x == hua - 1)) {
          double oldERadiusMax = eRadiusMax;
          eRadiusMax = eRadius[curJ][annInd++];
          System.out.println(" | " + eRadiusMax);
          assert annSet.stream().allMatch(a -> a > oldERadiusMax) : " some smaller than " + oldERadiusMax;
          assert annSet.stream().max((a, b) -> a.compareTo(b)).get().equals(eRadiusMax) : "None equal " + eRadiusMax;
          annSet.clear();
        }
        double val = cdist[curJ][curC[j]];
        System.out.print(" " + val + " ");
        annSet.add(val);
      }
      System.out.println(" | " + eRadius[curJ][annInd++]);
      System.out.println();
      return true;
    }

    /**
     * Finds the max value given by the indices between left and right. If j is
     * among the searched indices, it gets ignored.
     */
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

    private void selectSmallest(int[] indices, double values[], int right, int amount) {
      int left = 0;
      int goalIndex = amount - 1;
      while(true) {
        if(left == right) {
          return;
        }

        int pivotIndex = left + (int) (Math.random() * (right - left));
        pivotIndex = partition(indices, values, left, right, pivotIndex);

        if(pivotIndex == goalIndex) {
          return;
        }

        if(pivotIndex > goalIndex) {
          right = pivotIndex - 1;
        }
        else {
          left = pivotIndex + 1;
        }
      }
    }

    private int partition(int[] indices, double[] values, int left, int right, int pivotIndex) {
      double pivotValue = values[indices[pivotIndex]];
      swap(indices, pivotIndex, right);
      int storeIndex = left;
      for(int i = left; i < right; i++) {
        if(values[indices[i]] < pivotValue) {
          swap(indices, storeIndex, i);
          storeIndex++;
        }
      }
      swap(indices, right, storeIndex);
      return storeIndex;
    }

    /**
     * Swaps the elements arr[ind1] and arr[ind2];
     */
    private void swap(int[] arr, int ind1, int ind2) {
      int val1 = arr[ind1];
      arr[ind1] = arr[ind2];
      arr[ind2] = val1;
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
  public static class Par<V extends NumberVector> extends HamerlyKMeans.Par<V> {
    @Override
    public ExponionPartialSortKMeans<V> make() {
      return new ExponionPartialSortKMeans<>(distance, k, maxiter, initializer, varstat);
    }
  }
}
