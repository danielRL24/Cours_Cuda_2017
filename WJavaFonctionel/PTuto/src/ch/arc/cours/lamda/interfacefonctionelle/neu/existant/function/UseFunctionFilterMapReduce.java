
package ch.arc.cours.lamda.interfacefonctionelle.neu.existant.function;

import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Predicate;

import org.junit.Assert;

import ch.arc.cours.lamda.interfacefonctionelle.neu.existant.function.tools.Maison;
import ch.arc.cours.lamda.interfacefonctionelle.neu.existant.function.tools.MaisonTools;

public class UseFunctionFilterMapReduce
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
		System.out.println("filter map Reduce : Predicate Function binaryOperator ");

		filterMapReduce();
		}


	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	private static void filterMapReduce()
		{
		int n = 4;
		List<Maison> list = MaisonTools.create(n);

		int sum1 = version1(list);
		int sum2 = version2(list);
		int sum3 = version3(list);

		System.out.println(list);

		check(list, sum1);
		check(list, sum2);
		check(list, sum3);
		}

	/**
	 * somme des prix des maisons dont nbPiece>20
	 *
	 * Classe Interne anonyme
	 */
	private static int version1(List<Maison> list)
		{
		Predicate<Maison> filter = new Predicate<Maison>()
			{
				@Override
				public boolean test(Maison maison)
					{
					return maison.getNbPiece() > 20;
					}
			};

		Function<Maison, Integer> map = new Function<Maison, Integer>()
			{
				@Override
				public Integer apply(Maison maison)
					{
					return maison.getPrix();
					}
			};

		BinaryOperator<Integer> reduce = new BinaryOperator<Integer>()
			{
				@Override
				public Integer apply(Integer x, Integer y)
					{
					return x + y;
					}
			};

		return MaisonManipulator.filterMapReduce(list, filter, map, reduce, 0);
		}

	/*------------------------------*\
	|*				lamda			*|
	\*------------------------------*/

	/**
	 * somme des prix des maisons dont nbPiece>20
	 *
	 * lamda dans variable
	 */
	private static int version2(List<Maison> list)
		{
		Predicate<Maison> filter = maison -> maison.getNbPiece() > 20;
		Function<Maison, Integer> map = maison -> maison.getPrix();
		BinaryOperator<Integer> reduce = (x, y) -> x + y;

		return MaisonManipulator.filterMapReduce(list, filter, map, reduce, 0);
		}

	/**
	 * somme des prix des maisons dont nbPiece>20
	 *
	 * lamda
	 */
	private static int version3(List<Maison> list)
		{
		return MaisonManipulator.filterMapReduce(list, maison -> maison.getNbPiece() > 20, maison -> maison.getPrix(), (x, y) -> x + y, 0);
		}

	/*------------------------------*\
	|*				Tools			*|
	\*------------------------------*/

	private static void check(List<Maison> list, int sum)
		{
		System.out.println("sum prix = " + sum);

		Assert.assertTrue(sum == 7);
		}

	}
