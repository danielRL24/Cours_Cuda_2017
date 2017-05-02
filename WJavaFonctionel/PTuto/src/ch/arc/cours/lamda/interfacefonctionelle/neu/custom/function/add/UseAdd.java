
package ch.arc.cours.lamda.interfacefonctionelle.neu.custom.function.add;

import org.junit.Assert;

import ch.arc.cours.lamda.interfacefonctionelle.neu.custom.function.Function_I;

public class UseAdd
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
		System.out.println("Add");

		useAdd1();
		useAdd2();
		useAdd3();
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	/*------------------------------*\
	|*		add						*|
	\*------------------------------*/

	/**
	 * x -> (2*x)+ (x+1)
	 *
	 * h(x)= 2*x
	 * g(x)= x+1
	 * (g + h) (x)= h(x)+g(x)
	 *
	 * Classe interne anonyme
	 */
	private static void useAdd1()
		{
		Function_I h = new Function_I()
			{
			@Override
			public double value(double x)
				{
				return 2 * x;
				}
			};

		Function_I g = new Function_I()
			{
			@Override
			public double value(double x)
				{
				return x + 1;
				}
			};

		Function_I u1 = FunctionAdd.add1(h, g);
		Function_I u2 = FunctionAdd.add2(h, g);
		Function_I u3 = FunctionAdd.add3(h, g);

		check(u1, u2, u3);
		}

	/**
	 * x -> (2*x)+ (x+1)
	 *
	 * h(x)= 2*x
	 * g(x)= x+1
	 * (g + h) (x)= h(x)+g(x)
	 *
	 * lamda dans variable
	 */
	private static void useAdd2()
		{
		Function_I h = x -> 2 * x;
		Function_I g = x -> x + 1;

		Function_I u1 = FunctionAdd.add1(h, g);
		Function_I u2 = FunctionAdd.add2(h, g);
		Function_I u3 = FunctionAdd.add3(h, g);

		check(u1, u2, u3);
		}

	/**
	 * x -> (2*x)+ (x+1)
	 *
	 * h(x)= 2*x
	 * g(x)= x+1
	 * (g + h) (x)= h(x)+g(x)
	 *
	 * lamda
	 */
	private static void useAdd3()
		{
		Function_I u1 = FunctionAdd.add1(x -> 2 * x, x -> x + 1);
		Function_I u2 = FunctionAdd.add2(x -> 2 * x, x -> x + 1);
		Function_I u3 = FunctionAdd.add3(x -> 2 * x, x -> x + 1);

		check(u1, u2, u3);
		}

	/*------------------------------*\
	|*				tools			*|
	\*------------------------------*/

	private static void check(Function_I u1, Function_I u2, Function_I u3)
		{
		System.out.println(u1.value(2));
		System.out.println(u2.value(2));
		System.out.println(u3.value(2));

		Assert.assertTrue(u1.value(2) == (2 + 1) + (2 * 2));
		Assert.assertTrue(u2.value(2) == (2 + 1) + (2 * 2));
		Assert.assertTrue(u3.value(2) == (2 + 1) + (2 * 2));
		}

	}
