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
package elki.math;

import elki.data.NumberVector;
import elki.data.SparseNumberVector;

public class DotProduct {
  public static double dot(NumberVector v1, double[] v2) {
    if(v1 instanceof SparseNumberVector) {
      return sparseDot((SparseNumberVector) v1, v2);
    }
    return denseDot(v1, v2);
  }

  // TODO: implement for actual number vector
  public static double dot(NumberVector v1, NumberVector v2) {
    if(v1 instanceof SparseNumberVector) {
      return sparseDot((SparseNumberVector) v1, v2.toArray());
    }
    return denseDot(v1, v2.toArray());
  }

  public static double sparseDot(SparseNumberVector v1, double[] v2) {
    final int dim2 = v2.length;
    double dot = 0.;
    for(int i1 = v1.iter(); v1.iterValid(i1); i1 = v1.iterAdvance(i1)) {
      final int d1 = v1.iterDim(i1);
      if(d1 >= dim2) {
        break;
      }
      dot += v1.iterDoubleValue(i1) * v2[i1];
    }
    return dot;
  }

  public static double denseDot(NumberVector v1, double[] v2) {
    double dot = 0.;
    for(int i = 0; i < v1.getDimensionality(); i++) {
      dot += v1.doubleValue(i) * v2[i];
    }
    return dot;
  }
}
