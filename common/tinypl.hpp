/**
 * Tiny Parallel Library
 *
 * The library implements following parallel algorithms that its
 * interface are compatible with Intel TBB and Microsoft PPL.
 *  - parallel_for_each(first,last,func)
 *  - parallel_for(first,last,func)
 *  - parallel_invoke(f1,f2,...)  (up to 4 args)
 */
#ifndef TINYPL_HPP
#define TINYPL_HPP

#include <algorithm>
#include <iterator>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
//#define BOOST_ASIO_ENABLE_HANDLER_TRACKING
#include <boost/asio.hpp>

#ifndef TINYPL_WORKERNUM
#define TINYPL_WORKERNUM 0
#endif

#ifndef TINYPL_MIN_ITERATE
#define TINYPL_MIN_ITERATE 1
#endif


namespace tinypl
{
	namespace impl {

		// task scheduler
		class scheduler {
			friend struct waiter;
		public:
			explicit scheduler(std::size_t thnum = 0)
				: worker_(new boost::asio::io_service::work(iosrv_))
			{
				if (thnum == 0)
					thnum = std::max(1u, std::thread::hardware_concurrency());
				thnum_ = thnum;
				// start worker threads
				for (std::size_t i = 0; i < thnum - 1; ++i)
					thpool_.emplace_back(std::bind(static_cast<std::size_t(boost::asio::io_service::*)(void)>(&boost::asio::io_service::run), &iosrv_));
			}

			~scheduler()
			{
				// stop all worker threads
				worker_.reset();
				for (auto &th : thpool_)
					th.join();
			}

			// # of worker threads
			std::size_t worker_num() const { return thnum_; }

			// euqueue task
			template <class F>
			void enqueue(F f) { iosrv_.post(f); }

			std::vector<std::thread::id> get_thread_pool_id_list() const
			{
				std::vector<std::thread::id> list;
				list.reserve(thpool_.size());
				for (const auto &th : thpool_)
					list.push_back(th.get_id());

				return list;
			}

		public:
			// get scheduler object
			static scheduler& instance()
			{
				static scheduler sched(TINYPL_WORKERNUM);
				return sched;
			}

		private:
			boost::asio::io_service iosrv_;
			std::unique_ptr<boost::asio::io_service::work> worker_;
			std::vector<std::thread> thpool_;
			std::size_t thnum_;
		};

		// task waiter
		struct waiter {
			scheduler& sched_;
			volatile std::atomic<uint32_t> count_;

			waiter(scheduler& sched, unsigned int count)
				: sched_(sched), count_(count) {}
			~waiter()
			{
				while (0 < count_) {
					sched_.iosrv_.poll_one();
					// FIXME: It may cause heavy busyloop in worst-case scenario.
				}
			}

			struct holder {
				explicit holder(waiter& w) : w_(w) {}
				~holder() { w_.count_--; }
				waiter& w_;
			};
		};


		// task of parallel_for_each algorithm
		template <class Itr, class Func>
		void parallel_foreach_task(waiter* w, Itr first, Itr last, const Func& func)
		{
			waiter::holder h(*w);
			while (first != last)
				func(*first++);
		}

		// task of parallel_for algorithm
		template <class IdxType, class Func>
		void parallel_for_task(waiter* w, IdxType first, IdxType last, const Func& func)
		{
			waiter::holder h(*w);
			while (first < last)
				func(first++);
		}

		// task of parallel_invoke algorithm
		template <class Func>
		void parallel_invoke_task(waiter* w, const Func& func)
		{
			waiter::holder h(*w);
			func();
		}

	} // namespace impl


	/**
	 * parallel_for_each algorithm
	 */
	template <class Itr, class Func>
	void parallel_for_each(Itr first, Itr last, const Func& func)
	{
		impl::scheduler& sched = impl::scheduler::instance();
		std::size_t range = std::distance(first, last);
		std::size_t block = std::max(range / sched.worker_num(), std::size_t(TINYPL_MIN_ITERATE));
		impl::waiter w(sched, (range + block - 1) / block);
		for (Itr next = first; first != last; first = next) {
			std::advance(next, std::min(range, block));
			range -= std::min(range, block);
			if (next != last) {
				sched.enqueue(boost::bind(&impl::parallel_foreach_task<Itr, Func>, &w, first, next, func));
			}
			else {
				impl::parallel_foreach_task<Itr, Func>(&w, first, next, func);
			}
		}
	}

	/**
	* parallel_for algorithm
	*/
	template <class IdxType, class Func>
	void parallel_for(impl::scheduler& sched, IdxType first, IdxType last, const Func& func)
	{
		IdxType range = last - first;
		IdxType block = static_cast<IdxType>(std::max(range / sched.worker_num(), std::size_t(TINYPL_MIN_ITERATE)));
		impl::waiter w(sched, (range + block - 1) / block);
		for (IdxType next = first; first < last; first = next) {
			next = std::min(last, next + block);
			if (next < last) {
				sched.enqueue(std::bind(&impl::parallel_for_task<IdxType, Func>, &w, first, next, func));
			}
			else {
				impl::parallel_for_task<IdxType, Func>(&w, first, next, func);
			}
		}
	}

	/**
	* parallel_for algorithm
	*/
	template <class IdxType, class Func>
	void parallel_for(IdxType first, IdxType last, const Func& func)
	{
		impl::scheduler& sched = impl::scheduler::instance();
		parallel_for(sched, fist, last, func);
	}

	/**
	 * parallel_invoke algorithm (2 args)
	 */
	template <class F1, class F2>
	void parallel_invoke(const F1& f1, const F2& f2)
	{
		impl::scheduler& sched = impl::scheduler::instance();
		impl::waiter w(sched, 1);
		sched.enqueue(boost::bind(&impl::parallel_invoke_task<F1>, &w, f1));
		f2();
	}

	/**
	 * parallel_invoke algorithm (3 args)
	 */
	template <class F1, class F2, class F3>
	void parallel_invoke(const F1& f1, const F2& f2, const F3& f3)
	{
		impl::scheduler& sched = impl::scheduler::instance();
		impl::waiter w(sched, 2);
		sched.enqueue(boost::bind(&impl::parallel_invoke_task<F1>, &w, f1));
		sched.enqueue(boost::bind(&impl::parallel_invoke_task<F2>, &w, f2));
		f3();
	}

	/**
	 * parallel_invoke algorithm (4 args)
	 */
	template <class F1, class F2, class F3, class F4>
	void parallel_invoke(const F1& f1, const F2& f2, const F3& f3, const F4& f4)
	{
		impl::scheduler& sched = impl::scheduler::instance();
		impl::waiter w(sched, 3);
		sched.enqueue(boost::bind(&impl::parallel_invoke_task<F1>, &w, f1));
		sched.enqueue(boost::bind(&impl::parallel_invoke_task<F2>, &w, f2));
		sched.enqueue(boost::bind(&impl::parallel_invoke_task<F3>, &w, f3));
		f4();
	}

} // namespace tinypl

#endif
