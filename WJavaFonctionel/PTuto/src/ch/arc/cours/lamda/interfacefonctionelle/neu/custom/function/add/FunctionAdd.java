
package ch.arc.cours.lamda.interfacefonctionelle.neu.custom.function.add;

import ch.arc.cours.lamda.interfacefonctionelle.neu.custom.function.Function_I;

public class FunctionAdd
	{

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	/**
	 * (h + g)(x) = h(x)+g(x)
	 *
	 * classe interne anonyme
	 */
	public static Function_I add1(Function_I h, Function_I g)
		{
		return new Function_I()
			{
			@Override
			public double value(double x)
				{
				return h.value(x) + g.value(x);
				}
			};
		}

	/**
	 * (h + g)(x) = h(x)+g(x)
	 *
	 * lamda dans variable
	 */
	public static Function_I add2(Function_I h, Function_I g)
		{
		Function_I f = x -> h.value(x) + g.value(x);
		return f;
		}

	/**
	 * (h + g)(x) = h(x)+g(x)
	 *
	 * lamda
	 */
	public static Function_I add3(Function_I h, Function_I g)
		{
		return x -> h.value(x) + g.value(x);
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	}
