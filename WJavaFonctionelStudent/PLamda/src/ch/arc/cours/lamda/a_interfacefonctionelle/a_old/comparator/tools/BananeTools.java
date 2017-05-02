
package ch.arc.cours.lamda.a_interfacefonctionelle.a_old.comparator.tools;

import java.util.Arrays;
import java.util.Random;

public class BananeTools
	{

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	/*------------------------------*\
	|*			  Static			*|
	\*------------------------------*/

	public static Banane[] createTab(int n)
		{
		long seed = 1;
		Random random = new Random(seed);

		Banane[] tab = new Banane[n];
		for(int i = 0; i < n; i++)
			{
			tab[i] = new Banane(random.nextInt(10), random.nextInt(100));
			}

		return tab;
		}

	public static void print(String title, Banane[] tab)
		{
		System.out.println(title);
		System.out.println(Arrays.toString(tab));
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	}
