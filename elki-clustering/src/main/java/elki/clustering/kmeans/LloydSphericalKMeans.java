/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 * 
 * Copyright (C) 2020
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

import static elki.math.linearalgebra.VMath.timesEquals;

import java.util.Arrays;
import java.util.List;

import elki.clustering.kmeans.initialization.KMeansInitialization;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.KMeansModel;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDs;
import elki.database.ids.ModifiableDBIDs;
import elki.database.relation.Relation;
import elki.distance.NumberVectorDistance;
import elki.distance.UnitLengthEuclidianDistance;
import elki.logging.Logging;
import elki.math.DotProduct;
import elki.utilities.optionhandling.parameterization.Parameterization;

import net.jafama.FastMath;

public class LloydSphericalKMeans<V extends NumberVector> extends AbstractKMeans<V, KMeansModel> {
  private static final Logging LOG = Logging.getLogger(LloydSphericalKMeans.class);

  /**
   * Flag whether to compute the final variance statistic.
   */
  protected boolean varstat = true;

  public LloydSphericalKMeans(int k, int maxiter, KMeansInitialization initializer, boolean varstat) {
    super(UnitLengthEuclidianDistance.STATIC, k, maxiter, initializer);
    this.varstat = varstat;
  }

  @Override
  public Clustering<KMeansModel> run(Relation<V> relation) {
    Instance instance = new Instance(relation, distance, initialMeans(relation));
    instance.run(maxiter);
    return instance.buildResult(varstat, relation);
  }

  @Override
  protected Logging getLogger() {
    return LOG;
  }

  protected static class Instance extends AbstractKMeans.Instance {

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

    public Instance(Relation<? extends NumberVector> relation, NumberVectorDistance<?> df, double[][] means) {
      super(relation, df, means);
    }

    protected double similarity(NumberVector vec1, double[] means) {
      diststat++;
      return DotProduct.dot(vec1, means);
    }

    /**
     * Assign each object to the nearest cluster.
     *
     * @return number of objects reassigned
     */
    @Override
    protected int assignToNearestCluster() {
      assert k == means.length;
      int changed = 0;
      // Reset all clusters
      Arrays.fill(varsum, 0.);
      for(ModifiableDBIDs cluster : clusters) {
        cluster.clear();
      }
      for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
        NumberVector fv = relation.get(iditer);
        double maxSim = similarity(fv, means[0]);
        int maxIndex = 0;
        for(int i = 1; i < k; i++) {
          double sim = similarity(fv, means[i]);
          if(sim > maxSim) {
            maxIndex = i;
            maxSim = sim;
          }
        }
        varsum[maxIndex] += (1 - maxSim);
        clusters.get(maxIndex).add(iditer);
        if(assignment.putInt(iditer, maxIndex) != maxIndex) {
          ++changed;
        }
      }
      return changed;
    }

    @Override
    protected Logging getLogger() {
      return LOG;
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
        double length = .0;
        for(final double d : sum) {
          length += d * d;
        }
        length = FastMath.sqrt(length);
        newMeans[i] = timesEquals(sum, 1. / length);
      }
      return newMeans;
    }
  }

  /**
   * Parameterization class.
   *
   * @author Erich Schubert
   */
  public static class Par<V extends NumberVector> extends AbstractKMeans.Par<V> {
    @Override
    public void configure(Parameterization config) {
      super.configure(config);
      super.getParameterVarstat(config);
    }

    @Override
    public LloydSphericalKMeans<V> make() {
      return new LloydSphericalKMeans<V>(k, maxiter, initializer, varstat);
    }
  }
}
