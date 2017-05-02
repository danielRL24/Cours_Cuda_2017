
package ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.d_function.b_map.tools;

import java.util.ArrayList;
import java.util.List;

public class MaisonTools
	{

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	public static List<Maison> create(int n)
		{
		List<Maison> list = new ArrayList<Maison>(n);

		for(int i = 1; i <= n; i++)
			{
			int prix = i;
			int nbPiece = i*10;
			int surface = i*100;

			Maison maison = new Maison(prix,nbPiece, surface);

			list.add(maison);
			}

		return list;
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	}
