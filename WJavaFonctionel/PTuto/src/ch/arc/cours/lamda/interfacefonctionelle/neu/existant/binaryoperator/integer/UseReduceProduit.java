
package ch.arc.cours.lamda.interfacefonctionelle.neu.existant.binaryoperator.integer;

import java.util.Arrays;
import java.util.function.IntBinaryOperator;

import org.junit.Assert;

import ch.arc.cours.lamda.interfacefonctionelle.neu.existant.binaryoperator.NumberTools;

public class UseReduceProduit
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
		System.out.println("reduce : IntBinaryOperator : product");

		reduceProduit();
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	private static void reduceProduit()
		{
		int n = 4;
		int[] tab = NumberTools.create(n);

		int produit1 = version1(tab);
		int produit2 = version2(tab);
		int produit3 = version3(tab);

		System.out.println(Arrays.toString(tab));
		System.out.println("product = " + produit1);
		System.out.println("product = " + produit2);
		System.out.println("product = " + produit3);

		Assert.assertTrue(produit1 == 24);
		Assert.assertTrue(produit2 == 24);
		Assert.assertTrue(produit3 == 24);
		}

	/*------------------------------*\
	|*				produit			*|
	\*------------------------------*/

	/**
	 * classe interne anonyme
	 */
	private static int version1(int[] tab)
		{
		IntBinaryOperator operatorIntProduct = new IntBinaryOperator()
			{

			@Override
			public int applyAsInt(int x, int y)
				{
				return x * y;
				}
			};

		return NumberManipulatorInteger.reduce(tab, operatorIntProduct, 1);
		}

	/**
	 * lamda dans variable
	 */
	private static int version2(int[] tab)
		{
		IntBinaryOperator operatorIntProduct = (x, y) -> x * y;
		return NumberManipulatorInteger.reduce(tab, operatorIntProduct, 1);
		}

	/**
	 * lamda
	 */
	private static int version3(int[] tab)
		{
		return NumberManipulatorInteger.reduce(tab, (x, y) -> x * y, 1);
		}
	}
