
package ch.arc.cours.lamda.c_stream;

import java.util.List;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class UseStream_04_Infinite
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
		System.out.println("Stream Infinite");

		// stream infini
			{
			useInfinite();
			boucleByStream();
			}
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	/**
	 * Apprentissage
	 *
	 * 		stream pour afficher les 10 premiers nombres paires
	 */
	private static void useInfinite()
		{
		UnaryOperator<Integer> op = i -> i + 2;
		Stream<Integer> stream = Stream.iterate(0, op);

		List<Integer> list = stream.limit(10).collect(Collectors.toList());

		System.out.println(list);
		}

	/**
	 * Exercice:
	 *
	 * 		stream pour remplacer boucle
	 *
	 * 		rappel : stream.limite(n)
	 */
	private static void boucleByStream()
		{
		int n = 4;

		// code de base
			{
			for(int i = 0; i < n; i++)
				{
				System.out.println(i);
				}
			}

		// stream V1
			{
			UnaryOperator<Integer> op = i -> i + 1;
			Stream<Integer> stream = Stream.iterate(0, op);

			stream.limit(n).forEach(x -> System.out.println(x));
			}
		// stream V2
			{
			Stream.iterate(0, i -> i + 1).limit(n).forEach(System.out::println);
			}
		}

	}