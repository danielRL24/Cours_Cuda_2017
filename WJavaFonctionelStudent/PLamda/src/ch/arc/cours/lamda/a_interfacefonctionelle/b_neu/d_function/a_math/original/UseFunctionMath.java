
package ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.d_function.a_math.original;

import java.util.function.Function;

import org.junit.Assert;

public class UseFunctionMath
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
		System.out.println("math : Function");

		function();
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	private static void function()
		{
		add();

		compose();
		andThen();

		methodeReference();
		}

	/*------------------------------*\
	|*		Function<T1,T2>			*|
	\*------------------------------*/

	/**
	 * f(x) = (2*x)+ (x+1)
	 *
	 * h(x)= 2*x
	 * g(x)= x+1
	 * f(x) = (g + h) (x)= h(x)+g(x)
	 */
	private static void add()
		{
		//lamda dans variable
			{
			// TODO
			Function<Double,Double>	f=null;

			Assert.assertTrue(f.apply(2d) == 7);
			}
		}

	/**
	 * f(x) = x*x+1
	 *
	 * h(x)= x+1
	 * g(x)= x*x
	 * f(x) = (g o h) (x)= h(g(x))
	 */
	private static void compose()
		{
		// lamda dans variable
			{
			// TODO
			Function<Double,Double>	f=null;

			Assert.assertTrue(f.apply(2d) == 5);
			}
		}

	/**
	 * f(x) = 2(x+1)
	 */
	private static void andThen()
		{
		// lamda dans variable
			{
			// TODO
			Function<Double,Double>	f=null;

			Assert.assertTrue(f.apply(2d) == 6);
			}
		}

	/**
	 * f(x) = sqrt(cos(x))
	 */
	private static void methodeReference()
		{
		// lamda dans variable
			{
			// TODO
			Function<Double,Double>	f=null;

			Assert.assertTrue(isEquals(f.apply(2 * Math.PI), 1));
			}

		// lamda dans variable : reference methode
			{
			// TODO
			Function<Double,Double>	f=null;

			Assert.assertTrue(isEquals(f.apply(2 * Math.PI), 1));
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