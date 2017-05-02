
package ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.c_binaryoperator.b_reductionminmax;

import java.util.Comparator;
import java.util.List;
import java.util.function.BinaryOperator;

import ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.c_binaryoperator.b_reductionminmax.tools.Voiture;
import ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.c_binaryoperator.b_reductionminmax.tools.VoitureTools;

public class UseReduceMinMax
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
		System.out.println("reduce : BinaryOperator : min/ max");
		reduceMin();
		reduceMax();
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	private static void reduceMin()
		{
		int n = 3;
		List<Voiture> list = VoitureTools.create(n);

		// Warning : on veut pas le prix min, mais la voiture dont le prix est min
		Voiture voitureCheap = minPrix(list);

		System.out.println(list);
		System.out.println("Cheaper : " + voitureCheap);
		}

	private static void reduceMax()
		{
		int n = 3;
		List<Voiture> list = VoitureTools.create(n);

		// Warning : on veut pas le prix max, mais la voiture dont le prix est max
		Voiture voitureCheap = maxPrix(list);

		System.out.println(list);
		System.out.println("Expensive : " + voitureCheap);
		}

	/*------------------------------*\
	|*		BinaryOperator min		*|
	\*------------------------------*/

	/**
	 * Voiture dont le prix est minimimum
	 *
	 * lamda dans variable
	 *
	 * Warning : l'output n'est pas le prix de la voiture la moins cher, mais la voiture la moins cher!
	 */
	private static Voiture minPrix(List<Voiture> list)
		{
		// TODO
		Comparator<Voiture> comparator = null;
		BinaryOperator<Voiture> operator = BinaryOperator.minBy(comparator);
		return null;
		}

	/**
	 * Voiture dont le prix est minimimum
	 *
	 * lamda dans variable
	 *
	 * Warning : l'output n'est pas le prix de la voiture la plus cher, mais la voiture la plus cher!
	 */
	private static Voiture maxPrix(List<Voiture> list)
		{
		// TODO
		Comparator<Voiture> comparator =null;
		BinaryOperator<Voiture> operator = BinaryOperator.maxBy(comparator);
		return null;
		}

	}
