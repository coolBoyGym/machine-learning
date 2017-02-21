/*
C++代码实现基于一元线性模型的最小二乘法
参数1为输入文件，包含2n个间隔开的实数，前n个为x，后n个为对应的y
最小二乘法： Y = β1X + β2, 通过求导得到最佳的拟合直线
*/

#include<iostream>
#include<fstream>
#include<vector>
using namespace std;

class LeastSquare{
    double a, b;
public:
    LeastSquare(const vector<double>& x, const vector<double>& y){
        double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
        for(unsigned int i = 0; i < x.size(); ++i){
            t1 += x[i] * x[i];
            t2 += x[i];
            t3 += x[i] * y[i];
            t4 += y[i];
        }
        a = (t3 * x.size() - t2 * t4) / (t1 * x.size() - t2 * t2);  // 求得β1
        b = (t1 * t4 - t2 * t3) / (t1 * x.size() - t2 * t2);        // 求得β2
    }

    double getY(const double x) const { return a * x + b; }
    void print() const { cout << "y = " << a << "x + " << b << "\n"; }

};


int main(int argc, char *argv[])
{
    if(argc != 2){
        cout << "Usage: DataFile.txt" << endl;
        return -1;
    } else {
        vector<double> x;
        ifstream in(argv[1]);
        for(double d; in >> d; )
            x.push_back(d);
        int sz = x.size();
        vector<double> y(x.begin() + sz / 2, x.end());
        x.resize(sz / 2);
        LeastSquare ls(x, y);
        ls.print();

        cout << "Input x:" << endl;
        double x0;
        while(cin >> x0){
            cout << "y = " << ls.getY(x0) << endl;
            cout << "Input x:" << endl;
        }
    }
}

