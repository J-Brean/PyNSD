<div align="center">
  <h1>🍌PyNSD: The PNSD Toolkit🍌</h1>
  <img src="https://raw.githubusercontent.com/J-Brean/PyNPF/refs/heads/main/PNSDs.png" />
</div>

PyNSD is a Python-based graphical user interface designed for atmospheric scientists to process, visualise, and analyse Particle Number Size Distributions (PNSD). The toolkit streamlines the entire workflow from raw data (once you've turned it into a csv) to physical and statistical analyses, with a particular focus on New Particle Formation (NPF) events.

## Core Modules & Features
### Data Import & Processing
PyNSD features highly robust data handling capabilities designed for messy, real-world atmospheric measurements. Users can load standard time-series PNSD data (in CSV format) and seamlessly merge multiple, disparate datasets. The application automatically handles timestamp alignment, fills missing gaps, and allows users to apply custom data filtering to ensure only high-quality data progresses to the analytical stages.

### Summary Visualisations
The summary panel is for exploratory data analysis. It automatically generates contour plots (heatmaps) of the PNSD over time, overall average size distributions, average diurnal profiles, and time series of the total particle number concentration ($N_{tot}$). The user can zoom in on specific areas of their data, which in-turn updates the other plots.

### Temporal Trend Analysis
A dedicated module for exploring long-term trends in the PNSD. This module breaks down the dataset to investigate seasonal, day-of-week, and month-by-month trends.

### Manual NPF Identification & Physics
A day-by-day analysis tool where users can manually review daily contour plots and categorise them (e.g., NPF, Non-NPF, Undefined, Burst). For NPF days, users can calculate the GR using an interactive point-and-click tool or bounding boxes. The software calculates Condensation Sink (CS), Coagulation Sink, Growth Rate (GR), and Formation Rate ($J$).

### Deep Learning NPF Identification
PyPNSD incorporates a ResNet-50 Convolutional Neural Network as trained by Kecorius et al. (doi: 10.1038/s41597-024-04079-1). The algorithm automatically converts daily 1-hour resolution data into images, executes pattern recognition to identify 'banana-shaped' regional growth events, and assigns a confidence probability. The user can then subsequently calculate the GR from the identified NPF events.

### Cluster Analysis
A statistical module that allows the user to compare and then apply different clustering algorithms (such as K-means) directly to the size distribution spectra. This allows users to categorise the dataset into distinct, dominant aerosol states or background air mass types, providing insight into the most frequent atmospheric conditions at the measurement site.

### Nano Ranking
An evaluative tool which applies the nano-ranking framework of Aliaga et al. (doi: 10.5194/ar-1-81-2023), ranking days by the number of particles in a pre-defined bin.

## Future Roadmap
The toolkit is under active development, with the following modules planned for future releases:

Wind Sector Analysis: Vector-averaging of high-resolution wind speed and direction data to generate sector-specific diurnals, contours, and source-apportioned time series.

Positive Matrix Factorization (PMF): Advanced receptor modelling to identify and quantify the specific sources contributing to the measured aerosol distributions (PMF.exe not included)

## Installation
It is highly recommended to install PyNSD within a virtual environment to manage dependencies cleanly.

1. Clone the repository:

`git clone [https://github.com/YourUsername/PyNSD.git](https://github.com/YourUsername/PyNSD.git)
cd PyNSD`


2. Create and activate a virtual environment:

### Windows:

`python -m venv venv
venv\Scripts\activate`

### Mac/Linux:

`python3 -m venv venv
source venv/bin/activate`

3. Install the required dependencies:

`pip install -r requirements.txt`


## Usage

Once your virtual environment is active, simply launch the main application script:

python main.py


## Data Format

PyNSD expects time-series PNSD data in standard CSV format. The index must be a continuous datetime column, and the remaining headers must represent the geometric mean diameters of the measurement bins (in nanometres). There are some good tools for tidying up data that doesn't quite match.

## 👨‍🔬 Author

James Brean - Assistant Professor, University of Birmingham
j.brean@bham.ac.uk
