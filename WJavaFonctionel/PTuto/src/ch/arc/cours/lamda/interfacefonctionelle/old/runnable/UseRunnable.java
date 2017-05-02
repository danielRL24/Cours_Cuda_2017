
package ch.arc.cours.lamda.interfacefonctionelle.old.runnable;

public class UseRunnable
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
		version1();
		version2();
		version3();
		version4();

		// Version lamda
			{
			version5();
			version5Bis();
			version6();
			version6Bis();
			}
		}

	private static void version1()
		{
		Runnable runnable = new RunnableSeparer();

		Thread thread = new Thread(runnable);
		thread.start();
		}

	private static void version2()
		{
		Runnable runnable = new Runnable()
			{

			@Override public void run()
				{
				System.out.println("Hello2");
				}
			};

		Thread thread = new Thread(runnable);
		thread.start();
		}

	private static void version3()
		{
		Thread thread = new Thread(new Runnable()
			{

			@Override public void run()
				{
				System.out.println("Hello3");
				}
			});

		thread.start();
		}

	/**
	 * classe interne anonyme
	 * sans variable
	 */
	private static void version4()
		{
		new Thread(new Runnable()
			{
			@Override public void run()
				{
				System.out.println("Hello4");
				}
			}).start();
		}

	/*------------------------------*\
	|*		Version lamda			*|
	\*------------------------------*/

	/**
	 * Expression lamda, SYNTAXE FULL
	 */
	private static void version5()
		{
		new Thread(() -> {System.out.println("Hello5");}).start();
		}

	/**
	 * Expression lamda dans variable,SYNTAXE FULL
	 */
	private static void version5Bis()
		{
		Runnable runnable = () -> {System.out.println("Hello5bis");};
		new Thread(runnable).start();
		}

	/**
	 * Expression lamda, SYNTAXE LIGHT
	 */
	private static void version6()
		{
		new Thread(() -> System.out.println("Hello6")).start();
		}

	/**
	 * Expression lamda dans variable,SYNTAXE LIGHT
	 */
	private static void version6Bis()
		{
		Runnable runnable = () -> System.out.println("Hello6bis");
		new Thread(runnable).start();
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	}
