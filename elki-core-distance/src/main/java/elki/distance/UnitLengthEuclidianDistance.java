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

import elki.data.NumberVector;
import elki.math.DotProduct;

import net.jafama.FastMath;

/**
 * The euclidian distance can be calculated from the dot product. When both
 * vectors have unit length, it can be further simplified while preserving the
 * triangle inequality. This class computes this distance. <br>
 * 
 * @author alex
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
    return FastMath.sqrt(1 - DotProduct.dot(o1, o2));
  }

}
