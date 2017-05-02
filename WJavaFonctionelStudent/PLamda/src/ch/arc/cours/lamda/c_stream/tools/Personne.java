
package ch.arc.cours.lamda.c_stream.tools;

public class Personne
	{

	/*------------------------------------------------------------------*\
	|*							Constructeurs							*|
	\*------------------------------------------------------------------*/

	public Personne(int age, int hauteur)
		{
		this.age = age;
		this.hauteur = hauteur;
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	@Override
	public String toString()
		{
		StringBuilder builder = new StringBuilder();
		builder.append("Personne [age=");
		builder.append(this.age);
		builder.append(", hauteur=");
		builder.append(this.hauteur);
		builder.append("]");
		return builder.toString();
		}

	public void incAge()
		{
		this.age++;
		}

	/*------------------------------*\
	|*				Get				*|
	\*------------------------------*/

	public int getAge()
		{
		return this.age;
		}

	public int getHauteur()
		{
		return this.hauteur;
		}

	/*------------------------------*\
	|*				Set				*|
	\*------------------------------*/

	public void setAge(int age)
		{
		this.age = age;
		}

	public void setHauteur(int hauteur)
		{
		this.hauteur = hauteur;
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	/*------------------------------------------------------------------*\
	|*							Attributs Private						*|
	\*------------------------------------------------------------------*/

	// Inputs/Outputs
	private int age;
	private int hauteur;

	}
