import unittest
import numpy as np
import torch
import torch.nn.functional as F
from blast3r.inference.feature_depth import sample_feature_depth, reconstruct_feature_depth
from blast3r.inference.tracks import gather_data_from_dense_tracks
from blast3r.inference.rigs import combine_depth


class FeatureDepthTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.maps = {'base_log_depth': torch.randn(16, 20), 'features': torch.randn(15, 4, 5)}

    def test_grid_matches_epo_integer_coordinates(self):
        pixels = torch.tensor([[0.,0.], [19.,15.], [7.25,5.75]])
        grid = ((pixels + .5) / torch.tensor([20,16]) * 2 - 1)[None,None]
        expected = F.grid_sample(self.maps['features'][None], grid, align_corners=False,
                                 padding_mode='border')[0,:,0].T
        actual = sample_feature_depth(self.maps, pixels + .5)
        torch.testing.assert_close(actual[:, 2:], expected)
        self.assertEqual(actual[0,0], self.maps['base_log_depth'][0,0])

    def test_sparse_depth_matches_dense_output_at_all_pixel_centers(self):
        y,x = torch.meshgrid(torch.arange(16), torch.arange(20), indexing='ij')
        pixels = torch.stack((x+.5,y+.5),-1).reshape(-1,2)
        z = torch.randn(17)*.1
        z[0] = 1
        sparse = (sample_feature_depth(self.maps,pixels) @ z).exp().reshape(16,20)
        dense = reconstruct_feature_depth(self.maps,z)
        torch.testing.assert_close(sparse,dense,atol=1e-5,rtol=1e-5)
        numpy_maps = {k:v.numpy() for k,v in self.maps.items()}
        np.testing.assert_allclose(combine_depth(numpy_maps,z.numpy()), dense.numpy(),rtol=1e-6)

    def test_track_gathering_keeps_17_channels(self):
        views = [{'log_multidepth': self.maps}, {'log_multidepth': self.maps}]
        tracks = torch.tensor([[[.5,.5],[4.5,4.5]],[[10.5,7.5],[19.5,15.5]]])
        data = gather_data_from_dense_tracks(views,tracks)
        self.assertEqual(data.pix2d_dep.shape,(4,17))
        expected=torch.cat([sample_feature_depth(self.maps,tracks[:,i]) for i in range(2)])
        torch.testing.assert_close(data.pix2d_dep,expected)
        self.assertEqual(sample_feature_depth(self.maps,torch.empty(0,2)).shape,(0,17))


if __name__ == '__main__': unittest.main()
