/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>

#include "Camera.h"

namespace nl_uu_science_gmt
{

class Reconstructor
{
public:
	/*
	 * Voxel structure
	 * Represents a 3D pixel in the half space
	 */
	struct Voxel
	{
		int x, y, z;										 // Coordinates
		cv::Scalar color;									 // Color
		std::vector<cv::Point> camera_projection;			 // Projection location for camera[c]'s FoV (2D)
		std::vector<int> valid_camera_projection;			 // Flag if camera projection is in camera[c]'s FoV
		int label;											 // Labels
	};

private:
	const std::vector<Camera*> &m_cameras;					 // vector of pointers to cameras
	const int m_height;										 // Cube half-space height from floor to ceiling
	const int m_step;										 // Step size (space between voxels)

	std::vector<cv::Point3f*> m_corners;					 // Cube half-space corner locations
		
	size_t m_voxels_amount;									 // Voxel count
	cv::Size m_plane_size;									 // Camera FoV plane WxH
	
	std::vector<Voxel*> m_voxels;							 // Pointer vector to all voxels in the half-space
	std::vector<Voxel*> m_visible_voxels;					 // Pointer vector to all visible voxels	 
	
	cv::Mat m_cluster_labels;								 // Contains all labels of the visibile voxel vector 
	std::vector<std::vector<cv::Point2f>> m_cluster_centers; // Contains all cluster centers of the visible voxel vector
	std::vector<std::vector<int>> m_transform_vector;		 // Transformation vector to rectify the labels (derived from the Hungarian Algorithm)

	std::vector<cv::Mat> m_color_hists;				  		 // Contains all color histograms 
	std::vector<cv::Mat> m_reference_color_hists;			 // Contains all color histograms for the reference color model (frame 1)

	void initialize();

public:
	Reconstructor(
			const std::vector<Camera*> &);
	virtual ~Reconstructor();

	void update();
	void rectifyLabels();
	
	const std::vector<Voxel*>& getVisibleVoxels() const
	{
		return m_visible_voxels;
	}

	const std::vector<Voxel*>& getVoxels() const
	{
		return m_voxels;
	}

	const cv::Mat& getClusterLabels() const
	{
		return m_cluster_labels;
	}

	const std::vector<std::vector<cv::Point2f>>& getClusterCenters() const
	{
		return m_cluster_centers;
	}

		const std::vector<cv::Mat>& getColorHist() const
	{
		return m_color_hists;
	}

	const std::vector<cv::Mat>& getReferenceColorHist() const
	{
		return m_reference_color_hists;
	}

	void setVisibleVoxels(
			const std::vector<Voxel*>& visibleVoxels)
	{
		m_visible_voxels = visibleVoxels;
	}

	void setVoxels(
		const std::vector<Voxel*>& voxels)
	{
		m_voxels = voxels;
	}

	void setClusterLabels(
		const cv::Mat& clusterLabels)
	{
		m_cluster_labels = clusterLabels;
	}

	void setClusterCenters(
		const std::vector<std::vector<cv::Point2f>>& center)
	{
		m_cluster_centers = center;
	}

	void setColorHist(
		const std::vector<cv::Mat>& colorHist)
	{
		m_color_hists = colorHist;
	}

	void setReferenceColorHist(
		const std::vector<cv::Mat>& referenceColorHist)
	{
		m_reference_color_hists = referenceColorHist;
	}

	const std::vector<std::vector<int>>& getTransformVector() const
	{
		return m_transform_vector;
	}

	const std::vector<cv::Point3f*>& getCorners() const
	{
		return m_corners;
	}

	int getSize() const
	{
		return m_height;
	}

	const cv::Size& getPlaneSize() const
	{
		return m_plane_size;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
