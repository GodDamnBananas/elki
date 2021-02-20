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
import elki.distance.UnitLengthSphericalDistance;
import elki.logging.Logging;
import elki.utilities.optionhandling.parameterization.Parameterization;

/**
 * An unoptimized Spherical K-Means optimization based on
 * {@link HamerlyNoSKMeans}.
 * This version utilizes a distance Function that uses the dot
 * product for vectors of unit length and normalizes the means to unit length.
 * For more information, see
 * {@link UnitLengthSphericalDistance}
 * 
 * @author Alexander Voﬂ
 *
 * @param <V>
 */
public class HamerlyNoSUnoptimizedSphericalKMeans<V extends NumberVector> extends HamerlyNoSKMeans<V> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(HamerlyNoSUnoptimizedSphericalKMeans.class);

  public HamerlyNoSUnoptimizedSphericalKMeans(int k, int maxiter, KMeansInitialization initializer, boolean varstat) {
    super(UnitLengthSphericalDistance.STATIC, k, maxiter, initializer, varstat);
  }

  @Override
  public Clustering<KMeansModel> run(Relation<V> relation) {
    Instance instance = new Instance(relation, distance, initialMeans(relation));
    instance.run(maxiter);
    return instance.buildResult(varstat, relation);
  }

  /**
   * Inner instance, storing state for a single data set.
   */
  protected static class Instance extends HamerlyKMeans.Instance {

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
        dst[i] = UnitLengthSphericalDistance.normalize(sums[i]);
      }
    }

    @Override
    protected Logging getLogger() {
      return LOG;
    }

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
    public HamerlyNoSUnoptimizedSphericalKMeans<V> make() {
      return new HamerlyNoSUnoptimizedSphericalKMeans<>(k, maxiter, initializer, varstat);
    }
  }
}
