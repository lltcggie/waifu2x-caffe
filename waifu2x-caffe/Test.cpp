# include <iostream>
# include <fstream>
# include <iomanip>
# include <opencv2/dnn.hpp>
# include <opencv2/imgproc.hpp>
# include <opencv2/imgcodecs.hpp>

# include <opencv2/dnn/layer.details.hpp>
# include <opencv2/dnn/all_layers.hpp>

#define CV_VERSION_STR CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

// ビルドモード
#ifdef _DEBUG
#define CV_EXT_STR "d.lib"
#else
#define CV_EXT_STR ".lib"
#endif

#ifdef _MSC_VER

#pragma comment(lib, "opencv_core" CV_VERSION_STR CV_EXT_STR)
#pragma comment(lib, "opencv_imgcodecs" CV_VERSION_STR CV_EXT_STR)
#pragma comment(lib, "opencv_imgproc" CV_VERSION_STR CV_EXT_STR)
#pragma comment(lib, "opencv_dnn" CV_VERSION_STR CV_EXT_STR)
#pragma comment(lib, "libprotobuf" CV_EXT_STR)
#pragma comment(lib, "IlmImf" CV_EXT_STR)
#pragma comment(lib, "libjpeg-turbo" CV_EXT_STR)
#pragma comment(lib, "libopenjp2" CV_EXT_STR)
#pragma comment(lib, "libpng" CV_EXT_STR)
#pragma comment(lib, "libtiff" CV_EXT_STR)
#pragma comment(lib, "libwebp" CV_EXT_STR)
#pragma comment(lib, "zlib" CV_EXT_STR)

#pragma comment(lib, "cudart.lib")
//#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")

#endif

using namespace std;

class CropCenterLayer : public cv::dnn::Layer
{
private:
	std::vector<int> cropSize;
	cv::Ptr<cv::dnn::Layer> cropLayer;

public:
	CropCenterLayer(const cv::dnn::LayerParams& params) : Layer(params)
	{
		setParamsFrom(params);

		if (params.has("crop_size"))
		{
			const auto& paramCropSize = params.get("crop_size");
			const auto& str = paramCropSize.getStringValue();

			const int s = atoi(str.c_str());
			cropSize.resize(4);
			cropSize[0] = 0;
			cropSize[1] = 0;
			cropSize[2] = s;
			cropSize[3] = s;
		}

		cv::dnn::LayerParams parasm;
		parasm.set("axis", 0);
		parasm.set("offset", cv::dnn::DictValue::arrayInt(cropSize.data(), cropSize.size()));

		cropLayer = cv::dnn::CropLayer::create(parasm);
	}

	// Destructor
	virtual ~CropCenterLayer() = default;

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<CropCenterLayer>(new CropCenterLayer(params));
	}

	// Override virtual functions from cv::dnn::Layer and delegate to cropLayer

	CV_DEPRECATED_EXTERNAL virtual void finalize(const std::vector<cv::Mat*>& input, std::vector<cv::Mat>& output) override
	{
		cropLayer->finalize(input, output);
	}

	virtual void finalize(cv::InputArrayOfArrays inputs_arr, cv::OutputArrayOfArrays outputs_arr) override
	{
		std::vector<cv::Mat> inputs;
		inputs_arr.getMatVector(inputs);

		CV_Assert(inputs.size() == 1);
		const auto& input = inputs[0];

		//cv::Mat sizeShape(input.dims, input.size.p, input.type());
		cv::Mat sizeShape(input.size.dims(), input.size.p, input.type());

		auto& sz = sizeShape.size;
		for (int i = 0; i < sz.dims(); i++)
		{
			sz[i] = sz[i] - cropSize[i] * 2;
			CV_Assert(sz[i] >= 0);
		}

		inputs.push_back(sizeShape); // dummy second input for CropLayer

		cropLayer->finalize(inputs, outputs_arr);
	}

	CV_DEPRECATED_EXTERNAL virtual void forward(std::vector<cv::Mat*>& input, std::vector<cv::Mat>& output, std::vector<cv::Mat>& internals)
	{
		cropLayer->forward(input, output, internals);
	}

	virtual void forward(cv::InputArrayOfArrays inputs, cv::OutputArrayOfArrays outputs, cv::OutputArrayOfArrays internals) override
	{
		cropLayer->forward(inputs, outputs, internals);
	}

	virtual bool tryQuantize(const std::vector<std::vector<float>>& scales,
		const std::vector<std::vector<int>>& zeropoints, cv::dnn::LayerParams& params) override
	{
		return cropLayer->tryQuantize(scales, zeropoints, params);
	}

	CV_DEPRECATED_EXTERNAL void finalize(const std::vector<cv::Mat>& inputs, CV_OUT std::vector<cv::Mat>& outputs)
	{
		cropLayer->finalize(inputs, outputs);
	}

	CV_DEPRECATED std::vector<cv::Mat> finalize(const std::vector<cv::Mat>& inputs)
	{
		cropLayer->finalize(inputs);
	}

	CV_DEPRECATED CV_WRAP void run(const std::vector<cv::Mat>& inputs, CV_OUT std::vector<cv::Mat>& outputs,
		CV_IN_OUT std::vector<cv::Mat>& internals)
	{
		cropLayer->run(inputs, outputs, internals);
	}

	virtual int inputNameToIndex(cv::String inputName) override
	{
		return cropLayer->inputNameToIndex(inputName);
	}

	virtual int outputNameToIndex(const cv::String& outputName) override
	{
		return cropLayer->outputNameToIndex(outputName);
	}

	virtual bool supportBackend(int backendId) override
	{
		return cropLayer->supportBackend(backendId);
	}

	virtual cv::Ptr<cv::dnn::BackendNode> initHalide(const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& inputs) override
	{
		// preferableTargetを反映させるタイミングがここしかないっぽいので反映させる
		cropLayer->preferableTarget = preferableTarget;
		return cropLayer->initHalide(inputs);
	}

	virtual cv::Ptr<cv::dnn::BackendNode> initNgraph(const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& inputs,
		const std::vector<cv::Ptr<cv::dnn::BackendNode>>& nodes) override
	{
		// preferableTargetを反映させるタイミングがここしかないっぽいので反映させる
		cropLayer->preferableTarget = preferableTarget;
		return cropLayer->initNgraph(inputs, nodes);
	}

	virtual cv::Ptr<cv::dnn::BackendNode> initVkCom(const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& inputs,
		std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& outputs) override
	{
		// preferableTargetを反映させるタイミングがここしかないっぽいので反映させる
		cropLayer->preferableTarget = preferableTarget;
		return cropLayer->initVkCom(inputs, outputs);
	}

	virtual cv::Ptr<cv::dnn::BackendNode> initWebnn(const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& inputs,
		const std::vector<cv::Ptr<cv::dnn::BackendNode>>& nodes) override
	{
		// preferableTargetを反映させるタイミングがここしかないっぽいので反映させる
		cropLayer->preferableTarget = preferableTarget;
		return cropLayer->initWebnn(inputs, nodes);
	}

	virtual cv::Ptr<cv::dnn::BackendNode> initCUDA(void* context,
		const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& inputs,
		const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& outputs) override
	{
		// preferableTargetを反映させるタイミングがここしかないっぽいので反映させる
		cropLayer->preferableTarget = preferableTarget;
		return cropLayer->initCUDA(context, inputs, outputs);
	}

	virtual cv::Ptr<cv::dnn::BackendNode> initTimVX(void* timVxInfo,
		const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& inputsWrapper,
		const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& outputsWrapper,
		bool isLast) override
	{
		// preferableTargetを反映させるタイミングがここしかないっぽいので反映させる
		cropLayer->preferableTarget = preferableTarget;
		return cropLayer->initTimVX(timVxInfo, inputsWrapper, outputsWrapper, isLast);
	}

	virtual cv::Ptr<cv::dnn::BackendNode> initCann(const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& inputs,
		const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>& outputs,
		const std::vector<cv::Ptr<cv::dnn::BackendNode>>& nodes) override
	{
		// preferableTargetを反映させるタイミングがここしかないっぽいので反映させる
		cropLayer->preferableTarget = preferableTarget;
		return cropLayer->initCann(inputs, outputs, nodes);
	}

	virtual void applyHalideScheduler(cv::Ptr<cv::dnn::BackendNode>& node,
		const std::vector<cv::Mat*>& inputs,
		const std::vector<cv::Mat>& outputs,
		int targetId) const override
	{
		cropLayer->applyHalideScheduler(node, inputs, outputs, targetId);
	}

	virtual cv::Ptr<cv::dnn::BackendNode> tryAttach(const cv::Ptr<cv::dnn::BackendNode>& node) override
	{
		return cropLayer->tryAttach(node);
	}

	virtual bool setActivation(const cv::Ptr<cv::dnn::ActivationLayer>& layer) override
	{
		return cropLayer->setActivation(layer);
	}

	virtual bool tryFuse(cv::Ptr<cv::dnn::Layer>& top) override
	{
		return cropLayer->tryFuse(top);
	}

	virtual void getScaleShift(cv::Mat& scale, cv::Mat& shift) const override
	{
		cropLayer->getScaleShift(scale, shift);
	}

	virtual void getScaleZeropoint(float& scale, int& zeropoint) const override
	{
		cropLayer->getScaleZeropoint(scale, zeropoint);
	}

	virtual void unsetAttached() override
	{
		cropLayer->unsetAttached();
	}

	virtual bool getMemoryShapes(const std::vector<cv::dnn::MatShape>& inputs,
		const int requiredOutputs,
		std::vector<cv::dnn::MatShape>& outputs,
		std::vector<cv::dnn::MatShape>& internals) const override
	{
		CV_Assert(inputs.size() == 1);

		const auto& srcShape = inputs[0];

		std::vector<int> outShape(srcShape.size());
		for (int i = 0; i < srcShape.size(); i++)
		{
			outShape[i] = inputs[0][i] - cropSize[i] * 2;
		}
		outputs.assign(1, outShape);

		return false;

		//return cropLayer->getMemoryShapes(inputs, requiredOutputs, outputs, internals);
	}

	virtual bool updateMemoryShapes(const std::vector<cv::dnn::MatShape>& inputs) override
	{
		return cropLayer->updateMemoryShapes(inputs);
	}
};

static double sumAllElements(const cv::Mat& mat)
{
	CV_Assert(!mat.empty());

	const cv::Scalar s = cv::sum(mat); // チャンネルごとの合計
	double total = 0.0;
	for (int c = 0; c < mat.channels(); ++c) {
		total += s[c];
	}
	return total;
}


// ---- 内部実装 ----
template<typename T>
static void printRec(const cv::Mat& m, std::vector<int>& idx, int d) {
	if (d == m.dims - 1) {
		// 最終軸：一次元の並びを出力
		const int cn = m.channels();
		std::cout << "[";
		for (int i = 0; i < m.size[d]; ++i) {
			idx[d] = i;
			const T* p = m.ptr<T>(idx.data()); // idx の位置の要素先頭（ch=0）へのポインタ
			if (cn == 1) {
				std::cout << p[0];
			}
			else {
				std::cout << "(";
				for (int c = 0; c < cn; ++c) {
					std::cout << p[c];
					if (c + 1 < cn) std::cout << ", ";
				}
				std::cout << ")";
			}
			if (i + 1 < m.size[d]) std::cout << ", ";
		}
		std::cout << "]";
	}
	else {
		// 途中軸：再帰で内側へ
		std::cout << "[";
		for (int i = 0; i < m.size[d]; ++i) {
			idx[d] = i;
			printRec<T>(m, idx, d + 1);
			if (i + 1 < m.size[d]) std::cout << ",\n";
		}
		std::cout << "]";
	}
}

template<typename T>
static void printMatND_T(const cv::Mat& m) {
	// 浮動小数は小数桁を控えめに
	if (std::is_floating_point<T>::value) {
		std::cout << std::fixed << std::setprecision(6);
	}
	std::vector<int> idx(m.dims, 0);
	printRec<T>(m, idx, 0);
	std::cout << std::endl;
}

// エントリポイント（cv::Mat の depth に応じてディスパッチ）
static void printMatND(const cv::Mat& m) {
	switch (m.depth()) {
	case CV_8U:  printMatND_T<uchar>(m);   break;
	case CV_8S:  printMatND_T<schar>(m);   break;
	case CV_16U: printMatND_T<uint16_t>(m); break;
	case CV_16S: printMatND_T<int16_t>(m); break;
	case CV_32S: printMatND_T<int32_t>(m); break;
	case CV_32F: printMatND_T<float>(m);   break;
	case CV_64F: printMatND_T<double>(m);  break;
	default:
		throw std::runtime_error("Unsupported Mat depth.");
	}
}


void reg();
void reg2();

int main(int argc, char** argv) {
	//CV_DNN_REGISTER_LAYER_CLASS(CropCenter, CropCenterLayer);
	reg();
	reg2();

	// ImageNet Caffeリファレンスモデル
	string protoFile = "models/upresnet10/noise0_scale2.0x_model.prototxt";
	string modelFile = "models/upresnet10/noise0_scale2.0x_model.json.caffemodel";

	// 画像ファイル
	//string imageFile = (argc > 1) ? argv[1] : "images/cat.jpg";
	string imageFile = "red.png";

	// Caffeモデルの読み込み
	cv::dnn::Net net;
	try {
		net = cv::dnn::readNetFromCaffe(protoFile, modelFile);

		//net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		//net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	catch (const cv::Exception& e) {
		cerr << e.msg << endl;
		exit(-1);
	}

	// テスト用の入力画像ファイルの読み込み
	cv::Mat img = cv::imread(imageFile);
	if (img.empty()) {
		cerr << "can't read image: " << imageFile << endl;
		exit(-1);
	}
	try {
		// 入力画像をリサイズ
		int cropSize = 90;
		cv::resize(img, img, cv::Size(cropSize, cropSize));
		// Caffeで扱うBlob形式に変換 (実体はcv::Matのラッパークラス)
		const auto inputBlob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);

		//printMatND(inputBlob);

		std::vector<int> indim(inputBlob.size.p, inputBlob.size.p + inputBlob.size.dims());
		// 入力層に画像を入力
		net.setInput(inputBlob, "input");
		// フォワードパス(順伝播)の計算&出力層(Softmax)の出力を取得, ここに予測結果が格納されている
		// ImageNet 1000クラス毎の確率(32bits浮動小数点値)が格納された1x1000の行列(ベクトル)
		//const auto probMat = net.forward("/conv_post");
		const auto probMat = net.forward("/res1/axpy");

		std::vector<int> probMatDim(probMat.size.p, probMat.size.p + probMat.size.dims());
		auto sss = sumAllElements(probMat);
		//printMatND(probMat);

		std::vector<cv::Mat> outImgs;
		cv::dnn::imagesFromBlob(probMat, outImgs);
		//cv::dnn::imagesFromBlob(inputBlob, outImgs);
		auto outImg = outImgs[0];

		std::vector<int> outdim(outImg.size.p, outImg.size.p + outImg.size.dims());
		printMatND(outImg);

		//std::cout << cv::format(outImg, cv::Formatter::FMT_DEFAULT) << std::endl;

		// 値を0～1にクリッピング
		cv::threshold(outImg, outImg, 1.0, 1.0, cv::THRESH_TRUNC);
		cv::threshold(outImg, outImg, 0.0, 0.0, cv::THRESH_TOZERO);

		const double clip_eps8 = (1.0 / 255.0) * 0.5 - (1.0e-7 * (1.0 / 255.0) * 0.5);
		outImg.convertTo(outImg, CV_8U, 255.0, clip_eps8);

		cv::cvtColor(outImg, outImg, cv::COLOR_RGB2BGR);

		cv::imwrite("test.png", outImg);

		//// 確率(信頼度)の高い順にソートして、上位5つのインデックスを取得
		//cv::Mat sorted(probMat.rows, probMat.cols, CV_32F);
		//cv::sortIdx(probMat, sorted, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
		//cv::Mat topk = sorted(cv::Rect(0, 0, 5, 1));
		//// カテゴリ名のリストファイル(synset_words.txt)を読み込み
		//// データ例: categoryList[951] = "lemon";
		//vector<string> categoryList;
		//string category;
		//ifstream fs("synset_words.txt");
		//if (!fs.is_open()) {
		//	cerr << "can't read file" << endl;
		//	exit(-1);
		//}
		//while (getline(fs, category)) {
		//	if (category.length()) {
		//		categoryList.push_back(category.substr(category.find(' ') + 1));
		//	}
		//}
		//fs.close();
		//// 予測したカテゴリと確率(信頼度)を出力
		//cv::Mat_<int>::const_iterator it = topk.begin<int>();
		//while (it != topk.end<int>()) {
		//	cout << categoryList[*it] << " : " << probMat.at<float>(*it) * 100 << " %" << endl;
		//	++it;
		//}
	}
	catch (const cv::Exception& e) {
		cerr << e.msg << endl;
	}
	return 0;
}
