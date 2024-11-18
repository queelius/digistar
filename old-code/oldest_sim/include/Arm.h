#include "Vector.h"
#include "Point2.h"

namespace Alex
{
	struct Arm
	{
		static const size_t SIZE = 20;

		void initialize()
		{
			joint = Vector::zeros(SIZE);
			r = Vector::ones(SIZE);

			rom.resize(SIZE);
			for (size_t i = 1; i < SIZE; ++i)
				rom[i] = RoM(-PI / 2, PI / 2);

			p0 = Point2(0, 0);
		}

		struct RoM
		{
			RoM(double lower, double upper) :
			constraints(true), lower(lower), upper(upper) {};

			RoM() : constraints(false) {};

			double lower;
			double upper;
			bool constraints;

			bool valid(double x)
			{
				return !constraints || (x >= lower && x <= upper);
			};
		};

		bool articulate(const Vector& j)
		{
			for (size_t i = 0; i < SIZE; ++i)
			if (!rom[i].valid(joint[i] + j[i]))
				return false;

			Vector testJoint = joint;

			Vector jp = 0.001 * j;
			for (size_t i = 1; i <= 1000; ++i)
			{
				testJoint += jp;

				Vector p = Vector::zeros(SIZE);
				p[0] = testJoint[0];
				for (size_t i = 1; i < SIZE; ++i)
					p[i] = testJoint[i] + p[i - 1];

				std::vector<Point2> pts(SIZE + 1);
				pts[0] = p0;
				for (size_t i = 1; i <= SIZE; ++i)
					pts[i] = Point2(pts[i - 1].X() + r[i - 1] * cos(p[i - 1]), pts[i - 1].Y() + r[i - 1] * sin(p[i - 1]));

				std::vector<LineSegment> segs(SIZE);
				for (size_t i = 0; i < SIZE; ++i)
					segs[i] = LineSegment(pts[i + 1], pts[i]);

				Point2 tmp1, tmp2;
				for (size_t i = 0; i < SIZE - 2; ++i)
				{
					for (size_t j = i + 2; j < SIZE; ++j)
					{
						const auto x = segs[i].intersect(segs[j], tmp1, tmp2);
						if (x == INTERSECT || x == COLLINEAR_INTERSECT)
							return false;
					}
				}
			}

			joint += j;
			return true;
		}

		void draw()
		{
			Vector p = Vector::zeros(SIZE);
			p[0] = joint[0];
			for (size_t i = 1; i < SIZE; ++i)
				p[i] = joint[i] + p[i - 1];

			std::vector<Point2> pts(SIZE + 1);
			pts[0] = p0;
			for (size_t i = 1; i <= SIZE; ++i)
			{
				pts[i] = Point2(
					pts[i - 1].X() + r[i - 1] * cos(p[i - 1]),
					pts[i - 1].Y() + r[i - 1] * sin(p[i - 1])
					);
			}

			for (size_t i = 0; i < SIZE; ++i)
				drawLineSegment(LineSegment(pts[i + 1], pts[i]));
		}

		std::vector<RoM> rom;
		Vector joint;
		Vector r;
		Point2 p0;
	};
};