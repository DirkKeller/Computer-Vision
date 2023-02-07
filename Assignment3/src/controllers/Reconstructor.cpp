/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"
#include "Hungarian.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <cassert>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
	const vector<Camera*>& cs) :
	m_cameras(cs),
	m_height(2048),
	m_step(32)
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const float edgeL = m_height * 2, edgeM = m_height * 1.5, edgeS = m_height / 2;
	m_voxels_amount = ((edgeL + edgeS) / m_step) * ((edgeM + edgeS) / m_step) * (m_height / m_step); //(edge / m_step) * (edge / m_step) * (m_height / m_step);


	initialize();
}

/**
	* Deconstructor
	* Free the memory of the pointer vectors
	*/
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
	* Create some Look Up Tables
	* 	- LUT for the scene's box corners
	* 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
	* 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
	*/
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height * 2;
	const int xR = m_height / 2;
	const int yL = -m_height / 2;
	const int yR = m_height * 1.5;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(static) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	cout << "done!" << endl;
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	m_visible_voxels.clear();
	std::vector<Voxel*> visible_voxels;
	std::vector<cv::Point2f> groundCoordinates;

	int v;
#pragma omp parallel for schedule(static) private(v) shared(visible_voxels, groundCoordinates)
	for (v = 0; v < (int) m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}
#pragma omp critical
		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
			//push_back is critical 
			visible_voxels.push_back(voxel);
			groundCoordinates.push_back(Point2f(voxel->x, voxel->y));
		}
	}

	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());

	// ----------------------------------------------- OUR CODE: kmeans ------------------------------------
	
	// Hyperparameters
	const int clusters = 4;
	int conv_iter = 20;

	// Initialize output arrays
	std::vector<cv::Point2f> centers; 
	Mat labels; 

	// kmeans clusterin of the voxels collapsed on a 2D space (ground coordinates)
	double compactness = cv::kmeans(groundCoordinates, clusters, labels,
		TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 1.0),
		conv_iter, KMEANS_PP_CENTERS, centers);

	//cout << "Compactness: " << compactness << endl;

	m_cluster_labels = labels;
	m_cluster_centers.push_back(centers);

	// ----------------------------------------------- OUR CODE: retrieving the image pixels ------------------------------------

	// Initialization for reprojection
	Mat frame = m_cameras[1]->getFrame();
	std::vector<cv::Vec3b> person1, person2, person3, person4; // vectors for the histograms of each individual
	Mat img_person1(frame.rows, frame.cols, CV_8UC3, Scalar(0, 0, 0)), // image templates for colored display of each individuals
		img_person2 = img_person1.clone(),
		img_person3 = img_person1.clone(),
		img_person4 = img_person1.clone();

	// Voxel-based color reprojection
	int i;
	for (i = 0; i < visible_voxels.size(); i++) {

		// Retrieve image pixel values
		Point img_point = m_cameras[1]->projectOnView(Point3f(visible_voxels[i]->x, visible_voxels[i]->y, visible_voxels[i]->z));
		Vec3b BGR_pixel = frame.at<Vec3b>(img_point.y, img_point.x); // switched for reprojection (otherwise horizontal displayed)
		visible_voxels[i]->color = (Scalar)BGR_pixel;

		// Pushback image values and color histogram values for person 1 - 4
		if (labels.at<int>(i) == 0)
		{
			person1.push_back(BGR_pixel); // for histogram
			img_person1.at<Vec3b>(img_point) = frame.at<Vec3b>(img_point); // for image
		}
		else if (labels.at<int>(i) == 1)
		{
			person2.push_back(BGR_pixel); // for histogram
			img_person2.at<Vec3b>(img_point) = frame.at<Vec3b>(img_point); // for image
		}
		else if (labels.at<int>(i) == 2)
		{
			person3.push_back(BGR_pixel); // for histogram
			img_person3.at<Vec3b>(img_point) = frame.at<Vec3b>(img_point); // for image
		}
		else if (labels.at<int>(i) == 3)
		{
			person4.push_back(BGR_pixel); // for histogram
			img_person4.at<Vec3b>(img_point) = frame.at<Vec3b>(img_point); // for image
		}
	}

	// ----------------------------------------------- OUR CODE: color histograms ---------------------------------------------

	// Reduce noise in the color histograms by removing uniform pixel values (e.g. cut off the pants)
	visible_voxels.erase(std::remove_if(visible_voxels.begin(),
		visible_voxels.end(),
		[](decltype(visible_voxels)::value_type const& elem) {
			return elem->z < 100 || elem->z > 500;
		}),
		visible_voxels.end());

	// Initialize image and color histogram vector to loop through
	std::vector<std::vector<cv::Vec3b>> all_persons = { person1, person2, person3, person4 };
	std::vector<cv::Mat> color_hists, img_all_persons = { img_person1, img_person2, img_person3, img_person4 };

	// Configuring the histograms for each plane. 
	int histBinSize = 256; // maximal bin size is used due to better performance of cosine similarity
	float range[] = { 2, 256 }; // remove noise by removing the dark pixels
	const float* histRange[] = { range }; // set the range of values. Using B, G and R planes, values will range in the interval (0 - 255)
	bool uniform = true, accumulate = false;

	// Construct the color histograms for each person
	for (i = 0; i < (int)centers.size(); i++)
	{
		// Construct a color matrix per person
		vector<Mat> channels;
		Mat BGR_points(all_persons[i].size(), 1, CV_8UC3, all_persons[i].data()), hsv_image;
		cv::split(BGR_points, channels);

		//Construct the Color Histogram :
		Mat B_hist, G_hist, R_hist;
		cv::calcHist(&channels[0], 1, 0, Mat(), B_hist, 1, &histBinSize, histRange, uniform, accumulate);
		cv::calcHist(&channels[1], 1, 0, Mat(), G_hist, 1, &histBinSize, histRange, uniform, accumulate);
		cv::calcHist(&channels[2], 1, 0, Mat(), R_hist, 1, &histBinSize, histRange, uniform, accumulate);
		/*
	      &channels[0]: The source array(s)
					1 : The number of source arrays (here only 1).
					0 : The channel dimension to be measured. For BGR it is just the intensity (each array is single - channel), hence 0
				 Mat(): A mask to be used on the source array (zeros indicating pixels to be ignored; empty - not used defined here)
				_hist : The Mat object where the histogram will be stored
					1 : The histogram dimensionality
		  histBinSize : The number of bins per each used dimension
			histRange : The range of values to be measured per each dimension
			  uniform : The bin sizes are uniform distributed
		   accumulate : The histogram is cleared at the beginning
		*/

		// Create an image to display the histograms
		int hist_w = 512, hist_h = 640;
		int bin_w = cvRound((double)hist_w / histBinSize);
		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

		// Normalize the values of the histogram, such that they fall in the range indicated by the parameters entered:
		cv::normalize(B_hist, B_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		cv::normalize(G_hist, G_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		cv::normalize(R_hist, R_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		/*
					   _hist: Input array
					   _hist: Output normalized array
		0 and histImage.rows: The lower and upper limits to normalize the values of _hist
				 NORM_MINMAX: Argument that indicates the type of normalization (here, adjusting the values between the two limits set before)
						  -1: Implies that the output normalized array will be the same type as the input
		*/

		// Plot the individual color histograms
#pragma omp parallel for schedule(static) private(i) shared(histImage, bin_w, hist_h, B_hist, G_hist, R_hist)
		for (int i = 1; i < histBinSize; i++)
		{
			cv::line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(B_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(B_hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
			cv::line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(G_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(G_hist.at<float>(i))),
				Scalar(0, 255, 0), 2, 8, 0);
			cv::line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(R_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(R_hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
		}

		// Display the the color histograms and the reprojection of the voxel space on the camera
		cv::imshow("Individual", img_all_persons[i]);
		cv::imshow("Individual's Color Histogram", histImage);

		// Merge the channels and construct a global vector of all persons
		Mat HSV_hist;
		vector<Mat> HSV = { B_hist, G_hist, R_hist };
		merge(HSV, HSV_hist);
		color_hists.push_back(HSV_hist);
	}

	m_color_hists = color_hists;
}
// ------------------------------------ OUR CODE: rectify labels based on color histograms ----------------------------------------------
/**
 * Rectifies the labels assigned by kmeans based on the cosine similarity of the
 * color histogram signiture and the euclidian distance of the current and previous
 * location of the cluster centers. The Hungarian Algorithm is used for assignment.
 */
void Reconstructor::rectifyLabels()
{
	std::vector<cv::Mat> color_hists = m_color_hists, reference_color_hists = m_reference_color_hists;

	// Rectify labels after the first frame
	if (!reference_color_hists.empty())
	{
		int clusterSize = 4;
		int minCenterDist = 50; // threshold for minimal difference in euclidian distance

		int path_length = (int)m_cluster_centers.size() - 1;
		
		//cv::norm(peri[i] - pre[j]);

		std::vector<std::vector<double>> cosSimCandidates(clusterSize, vector<double>(clusterSize, 0));
		std::vector<std::vector<double>> euDistCandidates(clusterSize, vector<double>(clusterSize, 0));

		int i, j;
		for (i = 0; i < clusterSize; i++) // i = reference hist, previous center
		{
			cv::Mat refHist, newHist; // color 
			reference_color_hists[i].reshape(1, 1).convertTo(refHist, CV_64F); 

			std::vector<cv::Point2f> pre, peri; //position
			pre = m_cluster_centers[path_length - 1];

			for (j = 0; j < clusterSize; j++) // j = new hist, current center
			{
				color_hists[j].reshape(1, 1).convertTo(newHist, CV_64F); // color 
				peri = m_cluster_centers[path_length]; //position

				double cosSim = refHist.dot(newHist) / (cv::norm(refHist) * cv::norm(newHist)); // Cosine similarity of reference and new histogram		
				double euDist = cv::norm(peri[i] - pre[j]);

				cosSimCandidates[i][j] = (cosSim * -1) + 1; // revert the scale to cost
				euDistCandidates[i][j] = euDist;
			}
		}

		// Hungarian Algorithm for optimized (global) assignment
		vector<int> transform_vector_color, transform_vector_pos;
		HungarianAlgorithm HungAlgo;
		HungAlgo.Solve(cosSimCandidates, transform_vector_color); // color
		HungAlgo.Solve(euDistCandidates, transform_vector_pos); // position

		// Decision-making rules for position or color in case of conflict.
		if (transform_vector_color != transform_vector_pos)
		{
			int distCounter = 0;
			
			for (i = 0; i < clusterSize; i++)
			{
				vector<double>competitors = euDistCandidates[i];
				sort(competitors.begin(), competitors.end());
			
				if ((abs(competitors[0] - competitors[1])) > minCenterDist) distCounter++;
			}
			
			// Position is only used when cluster centers are not too close.
			if (distCounter == clusterSize) vector<int> transform_vector = transform_vector_pos;
			else vector<int> transform_vector = transform_vector_color;
		}	
		
		vector<int> transform_vector = transform_vector_color; // color-based rectification is default
		m_transform_vector.push_back(transform_vector_color);
	}
	else
	{
		m_transform_vector.push_back({ 0, 1, 2, 3 });
	}
}
// ---------------------------------- OUR CODE: rectify labels based on previous centers -------------------------------------------
} /* namespace nl_uu_science_gmt */