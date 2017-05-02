
package ch.arc.cours.lamda.interfacefonctionelle.neu.existant.binaryoperator.generic;

import java.util.function.BinaryOperator;

public class NumberManipulator
	{

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	/**
	 * see chapter stream to see beautiful flexible optimized code
	 */
	public static int reduce(int[] tab, BinaryOperator<Integer> operator, int initValue)
		{
		int value = initValue;
		for(int element:tab)
			{
			value = operator.apply(value, element);
			}

		return value;
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	}
