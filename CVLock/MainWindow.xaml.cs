using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Drawing;
using Microsoft.Kinect;
using Emgu.CV;
using Emgu.CV.Structure;
using System.IO;
using System.Windows.Threading;
using Emgu.CV.CvEnum;

namespace CVLock
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        KinectSensor sensor;
        WriteableBitmap depthBitmap;
        WriteableBitmap colorBitmap;
        DepthImagePixel[] depthPixels;
        byte[] colorPixels;


        /// <summary>
        /// Static password used for activation
        /// </summary>
        public static string password = "51324";

        /// <summary>
        /// Array to keep our current password in
        /// </summary>
        private char[] currentPass = new char[5];

        /// <summary>
        /// Index for password pointer
        /// </summary>
        private int index = 0;

        /// <summary>
        /// Numbero of blob countours detected
        /// </summary>
        int blobCount = 0;

        /// <summary>
        /// Simle counter to keep track of total images captured
        /// </summary>
        int trainPosCount = 0;

        /// <summary>
        /// Are we training the system?
        /// </summary>
        private bool isTraining = false;

        /// <summary>
        /// Are we training the system?
        /// </summary>
        private bool isTracking = false;

        
        
        /// <summary>
        /// Training box size
        /// </summary>
        private SizeF boxSize = new SizeF(200,200);

        /// <summary>
        /// Instructions for user
        /// </summary>
        private String[] trainMotivation = new string[7]{"Get Rady to place hand in box signaling a one","give me a two","give me a three","a four","and a five","done" , "thanks"};

        /// <summary>
        /// Timer used to create training data
        /// </summary>
        private DispatcherTimer scTimer;


        /// <summary>
        /// Eigen classifer with defult trianing data in //Images/positive/
        /// </summary>
        Classifier_Train Eigen_Recog = new Classifier_Train();

        /// <summary>
        /// Image that holds our tracked blobs in the countour
        /// </summary>
        Image<Gray, byte> result = null;

        /// <summary>
        /// Font on screen
        /// </summary>
        /// 
        MCvFont font = new MCvFont(FONT.CV_FONT_HERSHEY_COMPLEX, 1, 1);


        /// <summary>
        /// temp storage for last char recognized
        /// </summary>
        string lastChar = "";


        public MainWindow()
        {
            InitializeComponent();
           
            this.Loaded += MainWindow_Loaded;
            this.Closing += MainWindow_Closing;
            this.MouseDown += MainWindow_MouseDown;

        }

      
        void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            // initialize array with empty variables
            initializeArray();

            if (Eigen_Recog.IsTrained)
            {
                Console.WriteLine("Training Data loaded");
            }
            else
            {
                Console.WriteLine("No training data found, please train program using Train menu option");
            }

            foreach (var potentialSensor in KinectSensor.KinectSensors)
            {
                if (potentialSensor.Status == KinectStatus.Connected)
                {
                    this.sensor = potentialSensor;
                    break;
                }
            }

            //Sensor stuff start --
            if (null != this.sensor)
            {

                this.sensor.DepthStream.Enable(DepthImageFormat.Resolution640x480Fps30);
                this.sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
                this.colorPixels = new byte[this.sensor.ColorStream.FramePixelDataLength];
                this.depthPixels = new DepthImagePixel[this.sensor.DepthStream.FramePixelDataLength];
                this.colorBitmap = new WriteableBitmap(this.sensor.ColorStream.FrameWidth, this.sensor.ColorStream.FrameHeight, 96.0, 96.0, PixelFormats.Bgr32, null);
                this.depthBitmap = new WriteableBitmap(this.sensor.DepthStream.FrameWidth, this.sensor.DepthStream.FrameHeight, 96.0, 96.0, PixelFormats.Bgr32, null);                
                this.colorImg.Source = this.colorBitmap;

                this.sensor.AllFramesReady += this.sensor_AllFramesReady;

                try
                {
                    this.sensor.Start();
                }
                catch (IOException)
                {
                    this.sensor = null;
                }
            }

            if (null == this.sensor)
            {
                this.outputViewbox.Visibility = System.Windows.Visibility.Collapsed;
                this.txtError.Visibility = System.Windows.Visibility.Visible;
                this.txtInfo.Text = "No Kinect Found";
                
            }

            //Sensor Stuff end --

        }

        /// <summary>
        /// This is the frame event handler this method will run for evey frame captured , meat of app is here.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void sensor_AllFramesReady(object sender, AllFramesReadyEventArgs e)
        {
            BitmapSource depthBmp = null;
            blobCount = 0;
            int tempP = index % currentPass.Length;


            using (ColorImageFrame colorFrame = e.OpenColorImageFrame())
            {
                using (DepthImageFrame depthFrame = e.OpenDepthImageFrame())
                {
                    if (depthFrame != null)
                    {

                        blobCount = 0;

                        depthBmp = depthFrame.SliceDepthImage((int)sliderMin.Value, (int)sliderMax.Value);
                        
                        Image<Bgr, Byte> openCVImg = new Image<Bgr, byte>(depthBmp.ToBitmap());
                        Image<Gray, byte> gray_image = openCVImg.Convert<Gray, byte>();

                        using (MemStorage stor = new MemStorage())
                        {
                            //Find contours with no holes try CV_RETR_EXTERNAL to find holes
                            Contour<System.Drawing.Point> contours = gray_image.FindContours(
                             Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE,
                             Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_EXTERNAL,
                             stor);

                            for (int i = 0; contours != null; contours = contours.HNext)
                            {
                                i++;

                                //Get window for depth image
                                if ((contours.Area > Math.Pow(sliderMinSize.Value, 2)) && (contours.Area < Math.Pow(sliderMaxSize.Value, 2)))
                                {
                                    MCvBox2D box = contours.GetMinAreaRect();

                                    //Get box of countour
                                    System.Drawing.Rectangle roi2 = contours.BoundingRectangle;
                                    
                                    openCVImg.Draw(box, new Bgr(System.Drawing.Color.Red), 2);
                                    blobCount++;

                                    if (isTracking)
                                    {
                                        //Need another conversion to gray and bye here from rectangle
                                        result = gray_image.Copy(roi2).Convert<Gray, byte>().Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                                        this.toTrackIMG.Source = ImageHelpers.ToBitmapSource(result);
                                        if (Eigen_Recog.IsTrained)
                                        {
                                            string name = Eigen_Recog.Recognise(result);
                                            
                                            //Looks like we found a number add it to the password and make check if its correct
                                            if (!name.Equals("") && !name.Equals(lastChar))
                                            {
                                                currentPass[tempP] = (char)name[0];
                                                index++;
                                                ImageHelpers.saveBitMapToDisk(result.ToBitmap(), Convert.ToInt32(name), "Test", 1);
                                                displayCurrPass();
                                                
                                            }

                                            openCVImg.Draw(lastChar, ref font, new System.Drawing.Point((int)box.center.X, (int)box.center.Y), new Bgr(System.Drawing.Color.OrangeRed));
                                            if (!name.Equals(""))
                                            {
                                                //preserver the recognized symbole for future reference
                                                lastChar = name;
                                            }
                                        }
                                        
                                    }
                                    // Display bounding box for training sequence easy way to filter out noise
                                    else if (isTraining)
                                    {
                                        result = gray_image.Copy(roi2).Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                                        this.toTrackIMG.Source = ImageHelpers.ToBitmapSource(result);

                                    }
                                }
                            }
                            //Draw bounding box. Do this down here so that it's not mixed up with calculation of countours and eigen vectors
                            if (isTraining)
                            {
                                openCVImg = drawOnscreen(openCVImg);
                            }
                        }
                        //Draw on screen
                        this.outImg.Source = ImageHelpers.ToBitmapSource(openCVImg);                        
                        txtBlobCount.Text = blobCount.ToString();
                    }
                }
                //Draw color image 
                if (colorFrame != null)
                {
                    
                      colorFrame.CopyPixelDataTo(this.colorPixels);
                      this.colorBitmap.WritePixels(
                          new Int32Rect(0, 0, this.colorBitmap.PixelWidth, this.colorBitmap.PixelHeight),
                          this.colorPixels,
                          this.colorBitmap.PixelWidth * sizeof(int),
                          0);
                    
                }
            }
        }

        /// <summary>
        /// Used for drawing for hot corner boxes on screen
        /// </summary>
        /// <param name="img"></param>
        /// <returns></returns>
        public Image<Bgr, Byte> drawOnscreen(Image<Bgr, Byte> img)
        {
            //Assuming the image we're using is 640X480
            MCvBox2D boxUpR = new MCvBox2D(new PointF(200, 200), boxSize, 0);

            img.Draw(boxUpR, new Bgr(System.Drawing.Color.Green), 2);


            return img;
        }



        /// <summary>
        /// Collect training data for our gestures. This will simply create screen shotsof users hands in the requsted positions
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cTrainData(object sender, EventArgs e)
        {
            
            if (trainPosCount == 0)
            {
                this.feedback.Text = trainMotivation[trainPosCount];
            }
            else if (0 < trainPosCount && trainPosCount < trainMotivation.Length - 1)
            {
                this.feedback.Text = trainMotivation[trainPosCount];
                //Take 5 samples
                for (int i = 0; i < 5; ++i)
                {
                    ImageHelpers.saveBitMapToDisk(result.ToBitmap(), trainPosCount - 1, "positive",i);
                }
            }
            else
            {
                //We're done stop our training
                this.feedback.Text = trainMotivation[trainPosCount];
                stop_training();
            }

            trainPosCount++;

        }

        #region Password Methods
        /// <summary>
        /// Method used for validating super secure password!
        /// </summary>
        /// <param name="tocheck"></param>
        /// <returns></returns>
        private bool checkPass(char[] tocheck)
        {
            int count = 0;
            for (int i = 0; i < currentPass.Length; ++i)
            {
                if (tocheck[i].Equals(password[i]))
                {
                    count++;
                }
                Console.WriteLine("comparing passcode");
                Console.Write(tocheck[i]);
            }
            if (count < currentPass.Length)
            {
                return false;
            }
            return true;
        }

        /// <summary>
        /// Simple task to initialize our array with empty strings
        /// </summary>
        public void initializeArray()
        {
            for (int i = 0; i < password.Length; i++)
            {
                currentPass[i] = ' ';
            }
            index = 0;
        }

        /// <summary>
        /// Show the status of the current password on screen
        /// </summary>
        private void displayCurrPass()
        {
            this.passTxt.Text = new string(currentPass);

        }


        #endregion

        #region Window Stuff
        void MainWindow_MouseDown(object sender, MouseButtonEventArgs e)
        {
            this.DragMove();
        }


        void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (null != this.sensor)
            {
                this.sensor.Stop();
            }
        }

        private void CloseBtnClick(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
        #endregion

        #region Handle Buttons
        /// <summary>
        /// Handle begenning of training
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            if (isTraining)
            {
                this.feedback.Text = "Get ready";
                this.start_train_button.Content = "Start Training";
                stop_training();
                isTraining = false;
                trainPosCount = 0;
            }
            else
            {
                this.start_train_button.Content = "Stop Training";
                start_training();
                isTraining = true;
            }
        }
        /// <summary>
        /// Handle tracking buttonr
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Button_Click_2(object sender, RoutedEventArgs e)
        {
            if (isTracking)
            {
                //reset password variables
                initializeArray();
                index = 0;
                this.start_track_button.Content = "Start Tracking";
                isTracking = false;
            }
            else
            {
                this.start_track_button.Content = "Stop Tracking";

                isTracking = true;
            }
        }
        #endregion

        private void Button_Click_3(object sender, RoutedEventArgs e)
        {
            if (checkPass(currentPass))
            {
                this.passStatus.Text = "Granted";
                this.passStatus.Foreground = System.Windows.Media.Brushes.Green;
            }
            else
            {
                this.passStatus.Foreground = System.Windows.Media.Brushes.Red;
                this.passStatus.Text = "Denied";
            }
        }


        /// <summary>
        /// Start gathering images for training
        /// </summary>
        private void start_training()
        {

            scTimer = new DispatcherTimer();
               
            //attach event handler to timer
            scTimer.Tick += new EventHandler(cTrainData);

            //Run timer with 2 second intervals
            scTimer.Interval = new TimeSpan(0, 0, 3);
            scTimer.Start();


        }

        /// <summary>
        /// Stop gathering imagry
        /// </summary>
        private void stop_training()
        {
            scTimer.Stop();
            trainPosCount = 0;
            Eigen_Recog = new Classifier_Train();
        }

 
 

    }
}
