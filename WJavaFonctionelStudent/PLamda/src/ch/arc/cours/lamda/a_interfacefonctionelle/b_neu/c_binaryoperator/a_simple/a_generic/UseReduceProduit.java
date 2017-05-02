
package ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.c_binaryoperator.a_simple.a_generic;

import java.util.Arrays;

import org.junit.Assert;

import ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.c_binaryoperator.a_simple.NumberTools;

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
		System.out.println("reduce : BinaryOperator<Integer> : product");
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

		// check
			{
			System.out.println(Arrays.toString(tab));
			System.out.println("product = " + produit1);
			System.out.println("product = " + produit2);
			System.out.println("product = " + produit3);

			int produitTrue = 24;
			Assert.assertTrue(produit1 == produitTrue);
			Assert.assertTrue(produit2 == produitTrue);
			Assert.assertTrue(produit3 == produitTrue);
			}
		}

	/*------------------------------*\
	|*		BinaryOperator<T>		*|
	\*------------------------------*/

	/**
	 * classe interne anonyme
	 */
	private static int version1(int[] tab)
		{
		// TODO
		return -1;
		}

	/**
	 * lamda dans variable
	 */
	private static int version2(int[] tab)
		{
		// TODO
		return -1;
		}

	/**
	 * lamda
	 */
	private static int version3(int[] tab)
		{
		// TODO
		return -1;
		}

	}
