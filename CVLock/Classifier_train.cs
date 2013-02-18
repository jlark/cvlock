using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Structure;

namespace CVLock
{
    class Classifier_Train
    {
        #region Variables

        //Eigen
        MCvTermCriteria termCrit;
        EigenObjectRecognizer recognizer;

        //training variables
        List<Image<Gray, byte>> trainingImages = new List<Image<Gray, byte>>();//Images
        List<string> Names_List = new List<string>(); //labels
        int ContTrain, NumLabels;

        //Class Variables
        bool _IsTrained = false;
        string imagePath =  "../../Images/positive/";

        #endregion

        #region Constructors
        /// <summary>
        /// Default Constructor, Looks in (Application.StartupPath + "\\Images\Positive") for traing data.
        /// </summary>
        public Classifier_Train()
        {
            termCrit = new MCvTermCriteria(ContTrain, 0.001);
            _IsTrained = LoadTrainingData(imagePath);
        }

        /// <summary>
        /// Takes String input to a different location for training data
        /// </summary>
        /// <param name="Training_Folder"></param>
        public Classifier_Train(string Training_Folder)
        {
            termCrit = new MCvTermCriteria(ContTrain, 0.001);
            _IsTrained = LoadTrainingData(Training_Folder);
        }
        #endregion

        /// <summary>
        /// Recognise a Grayscale Image using the trained Eigen Recogniser
        /// </summary>
        /// <param name="Input_image"></param>
        /// <returns></returns>
        public string Recognise(Image<Gray, byte> Input_image)
        {
            if (_IsTrained)
            {
                string t = recognizer.Recognize(Input_image);
                return t;
            }
            else return "";//Blank prefered else can use null

        }

        /// <summary>
        /// <para>Return(True): If Training data has been located and Eigen Recogniser has been trained</para>
        /// <para>Return(False): If NO Training data has been located of error in training has occured</para>
        /// </summary>
        public bool IsTrained
        {
            get { return _IsTrained; }
        }

        /// <summary>
        /// Loads the traing data given a (string) folder location
        /// </summary>
        /// <param name="Folder_loacation"></param>
        /// <returns></returns>
        private bool LoadTrainingData(string Folder_loacation)
        {
            Names_List.Clear();
            trainingImages.Clear();

            //We should have 5 sets of trainig data
            for (int i = 0; i < 5; ++i)
            {
                int id = i + 1;
                for (int j = 0; j < 5; ++j)
                {
                    if (File.Exists(Folder_loacation + "\\hand_" + i + "_"+j+".png"))
                    {
                        //using index of 0 so let's increment i by one
                        Names_List.Add("" + id);
                        Console.WriteLine("Opening data " + i + " for training");
                        trainingImages.Add(new Image<Gray, byte>(Folder_loacation + "hand_" + i + "_"+j+".png"));

                    }
                    else return false;
                }
            }

            ContTrain = NumLabels;

            if (trainingImages.ToArray().Length != 0)
            {
                //Eigen face recognizer
                recognizer = new EigenObjectRecognizer(trainingImages.ToArray(),
                Names_List.ToArray(), 5000, ref termCrit); //5000 default
                return true;
            }
            else return false;

        }
    }
}
