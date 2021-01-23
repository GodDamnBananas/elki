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

import elki.clustering.kmeans.initialization.KMeansInitialization;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.KMeansModel;
import elki.database.relation.Relation;
import elki.distance.NumberVectorDistance;
import elki.distance.UnitLengthEuclidianDistance;
import elki.logging.Logging;

/**
 * An unoptimized Spherical K-Means optimization based on
 * {@link ExponionPartialSortKMeans}.
 * This version utilizes a distance Function that uses the dot
 * product for vectors of unit length and normalizes the means to unit length.
 * For more information, see
 * {@link UnitLengthEuclidianDistance}
 * 
 * @author Alexander Voﬂ
 *
 * @param <V>
 */
public class ExponionPartialSortUnoptimizedSphericalKMeans<V extends NumberVector> extends ExponionPartialSortKMeans<V> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(ExponionPartialSortUnoptimizedSphericalKMeans.class);

  /**
   * Constructor.
   *
   * @param k k parameter
   * @param maxiter Maxiter parameter
   * @param initializer Initialization method
   * @param varstat Compute the variance statistic
   */
  public ExponionPartialSortUnoptimizedSphericalKMeans(int k, int maxiter, KMeansInitialization initializer, boolean varstat) {
    super(UnitLengthEuclidianDistance.STATIC, k, maxiter, initializer, varstat);
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
   * @author Alexander Voﬂ
   */
  protected static class Instance extends ExponionPartialSortKMeans.Instance {

    /**
     * Constructor.
     *
     * @param relation Relation
     * @param means Initial means
     */
    public Instance(Relation<? extends NumberVector> relation, NumberVectorDistance<?> df, double[][] means) {
      super(relation, df, means);
    }

    /**
     * Compute means from cluster sums by averaging.
     * 
     * @param dst Output means
     * @param sums Input sums
     */
    @Override
    protected void meansFromSums(double[][] dst, double[][] sums) {
      for(int i = 0; i < k; i++) {
        dst[i] = UnitLengthEuclidianDistance.normalize(sums[i]);
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
  public static class Par<V extends NumberVector> extends ExponionPartialSortKMeans.Par<V> {
    @Override
    public ExponionPartialSortUnoptimizedSphericalKMeans<V> make() {
      return new ExponionPartialSortUnoptimizedSphericalKMeans<>(k, maxiter, initializer, varstat);
    }
  }
}
