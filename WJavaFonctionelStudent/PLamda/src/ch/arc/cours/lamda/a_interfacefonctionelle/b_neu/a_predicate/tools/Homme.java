
package ch.arc.cours.lamda.a_interfacefonctionelle.b_neu.a_predicate.tools;

public class Homme
	{

	/*------------------------------------------------------------------*\
	|*							Constructeurs							*|
	\*------------------------------------------------------------------*/

	public Homme(int poids, int hauteur)
		{
		this.poids = poids;
		this.hauteur = hauteur;
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	@Override
	public String toString()
		{
		StringBuilder builder = new StringBuilder();
		builder.append("Homme [poids=");
		builder.append(this.poids);
		builder.append(", hauteur=");
		builder.append(this.hauteur);
		builder.append("]");
		return builder.toString();
		}

	public void opposePoids()
		{
		this.poids*=-1;
		}

	/*------------------------------*\
	|*				Get				*|
	\*------------------------------*/

	public int getPoids()
		{
		return this.poids;
		}

	public int getHauteur()
		{
		return this.hauteur;
		}

	/*------------------------------*\
	|*				Set				*|
	\*------------------------------*/

	public void setPoids(int poids)
		{
		this.poids = poids;
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
	private int poids;
	private int hauteur;


	}
