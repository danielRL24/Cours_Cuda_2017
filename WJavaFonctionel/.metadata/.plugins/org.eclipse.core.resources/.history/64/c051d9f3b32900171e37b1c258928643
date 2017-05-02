
package ch.arc.cours.lamda.interfacefonctionelle.neu.custom.function.compose;

import ch.arc.cours.lamda.interfacefonctionelle.neu.custom.function.Function_I;

public class FunctionCompose
	{

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	/**
	 * (h 0 g)(x) = g(h(x))
	 *
	 * classe interne anonyme
	 */
	public static Function_I composition1(Function_I h, Function_I g)
		{
		return new Function_I()
			{
			@Override
			public double value(double x)
				{
				return h.value(g.value(x));
				}
			};
		}

	/**
	 * (h 0 g)(x) = g(h(x))
	 *
	 * lamda dans variable
	 */
	public static Function_I composition2(Function_I h, Function_I g)
		{
		Function_I f = x -> h.value(g.value(x));
		return f;
		}

	/**
	 * (h 0 g)(x) = g(h(x))
	 *
	 * lamda
	 */
	public static Function_I composition3(Function_I h, Function_I g)
		{
		return x -> h.value(g.value(x));
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	}
