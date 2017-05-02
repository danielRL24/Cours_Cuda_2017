
package ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.d_function.a_math.original;

import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.Assert;

public class UseSyntaxeExercice
	{

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	public static void main(String[] args)
		{
		main();
		}

	public static void main()
		{
		System.out.println("math : Function : exerciceSyntaxe");

		exerciceSyntaxe();
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	/*------------------------------*\
	|*		Function<T1,T2>			*|
	|*		BiFunction<T1,T2,T3>	*|
	\*------------------------------*/

	/**
	 * f= h op g
	 * f(x) = h(x) op g(x)
	 *
	 * examples:
	 * 		op = +
	 * 		op = *
	 */
	private static <T1, T2> Function<T1, T2> op(Function<T1, T2> h, Function<T1, T2> g, BiFunction<T2, T2, T2> op)
		{
		// lamda
			{
			// TODO
			return null;
			}
		}

	private static Function<Double, Double> add(Function<Double, Double> h, Function<Double, Double> g)
		{
		// lamda dans variable
			{
			// TODO
			}

		// lamda
			{
			// TODO
			return null;
			}
		}

	private static Function<Double, Double> prod(Function<Double, Double> h, Function<Double, Double> g)
		{
		// lamda dans variable
			{
			// TODO
			}

		// lamda
			{
			// TODO
			return null;
			}
		}

	/**
	 * f(x) = cos(x)*cos(x)+sin(x)*sin(x)
	 */
	private static void exerciceSyntaxe()
		{
		// lamda dans variable
			{
			// TODO
			Function<Double, Double> cos = null;
			Function<Double, Double> sin =null;

			Function<Double, Double> cos2 =null;
			Function<Double, Double> sin2 =null;
			Function<Double, Double> f =null;

			Assert.assertTrue(f.apply(Math.PI) == 1);
			Assert.assertTrue(f.apply(Math.PI / 2) == 1);
			}

		// lamda
			{
			// TODO
			Function<Double, Double> cos =null;
			Function<Double, Double> sin =null;

			Function<Double, Double> f =null;

			Assert.assertTrue(isEquals(f.apply(Math.PI), 1));
			Assert.assertTrue(isEquals(f.apply(Math.PI / 2), 1));
			}
		}

	/*------------------------------*\
	|*				tools			*|
	\*------------------------------*/

	private static boolean isEquals(double x1, double x2)
		{
		final double epsilon = 1e-6;
		return Math.abs(x1 - x2) < epsilon;
		}

	}
