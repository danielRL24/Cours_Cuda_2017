
package ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.c_binaryoperator.a_simple;

public class NumberTools
	{

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	/**
	 * [1,n]
	 */
	public static int[] create(int n)
		{
		int[] tab = new int[n];

		for(int i = 0; i < n; i++)
			{
			tab[i] = i + 1;
			}

		return tab;
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	}
