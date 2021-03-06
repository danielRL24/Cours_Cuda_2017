
package ch.arc.cours.lamda.b_foreach;

import java.io.Serializable;

public class Int implements Serializable
	{

	/*------------------------------------------------------------------*\
	|*							Constructeurs							*|
	\*------------------------------------------------------------------*/

	public Int(int i)
		{
		this.i = i;
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	@Override
	public String toString()
		{
		return EMPTY + i;
		}

	public int inc()
		{
		i++;
		return i;
		}

	/*------------------------------*\
	|*				Get				*|
	\*------------------------------*/

	public int intvalue()
		{
		return this.i;
		}

	/*------------------------------*\
	|*				Set				*|
	\*------------------------------*/

	public void setValue(int i)
		{
		this.i = i;
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	/*------------------------------------------------------------------*\
	|*							Attributs Private						*|
	\*------------------------------------------------------------------*/

	private int i;

	/*------------------------------*\
	|*			  Static			*|
	\*------------------------------*/

	private static final String EMPTY = "";

	}
