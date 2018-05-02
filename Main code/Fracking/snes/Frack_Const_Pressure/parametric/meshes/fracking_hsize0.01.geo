
			lc = DefineNumber[ 0.01, Name "Parameters/lc" ];
			H = 4;
			L = 4;
		        Point(1) = {0, 0, 0, 10*lc};
		        Point(2) = {L, 0, 0, 10*lc};
		        Point(3) = {L, H, 0, 10*lc};
		        Point(4) = {0, H, 0, 10*lc};
		        Point(5) = {1.8, H/2, 0, 1*lc};
		        Point(6) = {2.2, H/2, 0, 1*lc};
		        Line(1) = {1, 2};
		        Line(2) = {2, 3};
		        Line(3) = {3, 4};
		        Line(4) = {4, 1};
		        Line Loop(5) = {1, 2, 3, 4};
			Plane Surface(30) = {5};

			Line(6) = {5, 6};
		        Line{6} In Surface{30};


			Physical Surface(1) = {30};

			Physical Line(101) = {6};

	