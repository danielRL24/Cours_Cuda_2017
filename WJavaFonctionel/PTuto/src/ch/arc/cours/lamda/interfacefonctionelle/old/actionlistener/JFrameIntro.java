
package ch.arc.cours.lamda.interfacefonctionelle.old.actionlistener;

import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;

public class JFrameIntro extends JFrame
	{

	/*------------------------------------------------------------------*\
	|*							Constructeurs							*|
	\*------------------------------------------------------------------*/

	public JFrameIntro()
		{
		geometry();
		control();
		appearance();
		}

	/*------------------------------------------------------------------*\
	|*							Methodes Public							*|
	\*------------------------------------------------------------------*/

	/*------------------------------------------------------------------*\
	|*							Methodes Private						*|
	\*------------------------------------------------------------------*/

	private void geometry()
		{
		// JComponent : Instanciation
		button0 = new JButton("0");
		button0Bis = new JButton("0 Bis");
		button1 = new JButton("1");
		button2 = new JButton("2");
		button3 = new JButton("3");
		button3Bis = new JButton("3 bis");
		button4 = new JButton("4");
		button4Bis = new JButton("4 bis");
		button5 = new JButton("5");
		button5Bis = new JButton("5 bis");
		button6 = new JButton("6");
		button6Bis = new JButton("6 bis");

		// Layout : Specification
			{
			FlowLayout layout = new FlowLayout();
			setLayout(layout);
			}

		// JComponent : add
		add(button0);
		add(button0Bis);
		add(button1);
		add(button2);
		add(button3);
		add(button3Bis);
		add(button4);
		add(button4Bis);
		add(button5);
		add(button5Bis);
		add(button6);
		add(button6Bis);
		}

	private void control()
		{
		setDefaultCloseOperation(EXIT_ON_CLOSE);

		// Version 0
			{
			ActionListener actionListener = new ActionListenerSeparer();
			button0.addActionListener(actionListener);
			}

		// Version 0 bis
			{
			button0Bis.addActionListener(new ActionListenerSeparer());
			}

		// Version 1
			{
			ActionListener actionListener = new ActionListener()
				{

				@Override
				public void actionPerformed(ActionEvent e)
					{
					System.out.println("click 1");
					}
				};

			button1.addActionListener(actionListener);
			}

		// Version 2
			{
			button2.addActionListener(new ActionListener()
				{

				@Override
				public void actionPerformed(ActionEvent evt)
					{
					System.out.println("click 2");
					}
				});
			}

		// Versions expression lamda
			{
			// version3 (avec typage)
				{
				button3.addActionListener((ActionEvent evt) ->{System.out.println("click 3");});

				// dans Variable
					{
					ActionListener actionListener = (ActionEvent evt) ->{System.out.println("click 3 bis");};

					button3Bis.addActionListener(actionListener);
					}
				}

			// verison 4 (sans typage)
				{
				button4.addActionListener((evt) -> {System.out.println("click 4");});

				// dans Variable
					{
					ActionListener actionListener = (ActionEvent evt) -> System.out.println("click 4 bis");

					button4Bis.addActionListener(actionListener);
					}
				}

			// Version 5 (sans typage light)
				{
				button5.addActionListener(evt ->{System.out.println("click 5");});

				// dans Variable
					{
					ActionListener actionListener = (evt) -> {System.out.println("click 5 bis");};

					button5Bis.addActionListener(actionListener);
					}
				}

			// Version 6 (super light)
				{
				button6.addActionListener(evt -> System.out.println("click 6"));

				// dans Variable
					{
					ActionListener actionListener = evt -> System.out.println("click 6 bis");

					button6Bis.addActionListener(actionListener);
					}
				}
			}
		}

	private void appearance()
		{
		setSize(600, 400);
		setLocationRelativeTo(null); // frame centrer
		setVisible(true); // last!
		}

	/*------------------------------------------------------------------*\
	|*							Attributs Private						*|
	\*------------------------------------------------------------------*/

	// Tools
	private JButton button0;
	private JButton button0Bis;
	private JButton button1;
	private JButton button2;
	private JButton button3;
	private JButton button3Bis;
	private JButton button4;
	private JButton button4Bis;
	private JButton button5;
	private JButton button5Bis;
	private JButton button6;
	private JButton button6Bis;

	}
