## Requirements

	numpy >= 1.14.2
	scikit-learn >= 0.19.1
	plotly >= 2.5.1

## Dataset

Generate complex synthetic dataset. 

- Data generator functions are in soydata.data
- Visualization functions are in soydata.visualize

	from soydata.data import *
	from soydata.visualize import *

### Two moon

	X, color = make_moons(n_samples=300, 
	    xy_ratio=2.0, x_gap=-.2, y_gap=0.2, noise=0.1)

	ipython_2d_scatter(X, color)

![](./figures/soydata_two_moon.png)

### Spiral

	X, color = make_spiral(n_samples_per_class=500, n_classes=3,
	    n_rotations=3, gap_between_spiral=0.1, 
	    gap_between_start_point=0.1, equal_interval=True,                
	    noise=0.2)

	ipython_2d_scatter(X, color)

![](./figures/soydata_spiral.png)

### Swiss roll

	X, color = make_swiss_roll(n_samples=3000, n_rotations=3, 
	    gap=0.5, thickness=0.0, width=10.0)

	ipython_3d_scatter(X, color)

![](./figures/soydata_swissroll.png)

### Radial

	X, color = make_radial(n_samples_per_sections=100, n_classes=2, 
	    n_sections_per_class=3, gap=0.1, equal_proportion=True,
	    radius_min=0.1, radius_base=1.0, radius_variance=0.5)

	ipython_2d_scatter(X, color)

![](./figures/soydata_radal.png)

### Two layer radial

	X, color = make_two_layer_radial(n_samples_per_sections=100, n_classes=2, 
	    n_sections_per_class=3, gap=0.0, equal_proportion=False)

	ipython_2d_scatter(X, color)

![](./figures/soydata_two_layer_radial.png)

### Rectangular

	X, color = make_rectangular(n_samples=500, 
	    label=0, x_b=0, x_e=10, y_b=0, y_e=10)

	ipython_2d_scatter(X, color)

![](./figures/soydata_rectangular.png)

### Triangular

Upper triangular

	X, color = make_triangular(n_samples=500, upper=True,
	    label=0, x_b=0, x_e=10, y_b=0, y_e=10)

	ipython_2d_scatter(X, color)

![](./figures/soydata_upper_triangular.png)

Lower triangular

	X, color = make_triangular(n_samples=500, upper=False,
	    label=0, x_b=0, x_e=10, y_b=0, y_e=10)

	ipython_2d_scatter(X, color)

![](./figures/soydata_lower_triangular.png)

### Decision Tree dataset 1

	X, color = get_decision_tree_data_1(n_samples=2000)
	ipython_2d_scatter(X, color)

![](./figures/soydata_decision_tree1.png)

### Decision Tree dataset 2

	X, color = get_decision_tree_data_2(n_samples=2000)
	ipython_2d_scatter(X, color)

![](./figures/soydata_decision_tree2.png)
