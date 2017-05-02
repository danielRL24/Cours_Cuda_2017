
package ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.b_consumer;

import java.util.function.Consumer;
import java.util.function.Predicate;

import ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.a_predicate.tools.Homme;

public class HommeManipulator
	{

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	/**
	 * sur place, modifie homme
	 *
	 * see chapter stream to see beautiful flexible optimized code
	 */
	public static void foreach(Iterable<Homme> iterable, Consumer<Homme> consumer)
		{
		for(Homme homme:iterable)
			{
			// TODO
			}

		}

	/**
	 * sur place, modifie homme filtrer
	 *
	 * see chapter stream to see beautiful flexible optimized code
	 */
	public static void filterForeach(Iterable<Homme> iterable, Predicate<Homme> predicate, Consumer<Homme> consumer)
		{
		for(Homme homme:iterable)
			{
			// TODO
			}
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	}
