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
package elki.distance;

import static elki.math.linearalgebra.VMath.timesEquals;

import java.util.Arrays;

import elki.data.NumberVector;
import elki.math.DotProduct;

import net.jafama.FastMath;

/**
 * This class computes the metric distance d(A,B) = sqrt(1-A*B) which is derived
 * from the Euclidian Distance. <br>
 * 
 * ||A-B|| = sqrt(2(1-A*B)) = sqrt(2)*sqrt(1-A*B), therefore sqrt(1-A*B) is a
 * metric if A and B are
 * normalized to unit length.
 * 
 * @author Alexander Voﬂ
 *
 */
public class UnitLengthEuclidianDistance extends AbstractNumberVectorDistance {

  /**
   * Static instance. Use this!
   */
  public static final UnitLengthEuclidianDistance STATIC = new UnitLengthEuclidianDistance();

  /**
   * Constructor - use {@link #STATIC} instead.
   * 
   * @deprecated Use static instance!
   */
  @Deprecated
  public UnitLengthEuclidianDistance() {
    super();
  }

  @Override
  public boolean isMetric() {
    return true;
  }

  @Override
  public double distance(NumberVector o1, NumberVector o2) {
    return FastMath.sqrt(Math.max(0, 1 - DotProduct.dot(o1, o2)));
  }

  /**
   * Returns a copy of given vector that is normalized to unit length.
   * 
   * @param vec vector as double array
   * @return normalized copy of given vector
   */
  public static double[] normalize(double[] vec) {
    double length = .0;
    for(final double d : vec) {
      length += d * d;
    }
    length = FastMath.sqrt(length);
    return timesEquals(Arrays.copyOf(vec, vec.length), 1. / length);
  }

}
