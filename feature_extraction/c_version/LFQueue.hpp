#include <atomic>
#include <thread>

class ConcurrencyError : public std::exception {
private:
	std::string mWhat;
public:
	ConcurrencyError(const char *vcpError) : exception(), mWhat(vcpError) {}
	~ConcurrencyError() throw() {}

	virtual const char* what() const throw() {
		return mWhat.c_str();
	}
};

template <typename T> class LFQueue1P1C {
private:
	struct Node {
		Node(T vValue) : mValue(vValue), pNext(nullptr), m_bValid(false) { }
		T        mValue;
		bool	 m_bValid;
		Node    *pNext;

		bool operator!=(const Node &node) const {
			return this != &node;
		}
	};

	Node                *pFirst;    // Producer Only
	std::atomic<Node *> pDivider;   // Producer / Consumer
	std::atomic<Node *> pLast;      // Producer / Consumer
	std::thread::id     producer;
	std::thread::id     consumer;

public:
	// This queue must be fully constructed before being used in another thread.
	LFQueue1P1C(std::thread::id producer = std::thread::id(), std::thread::id consumer = std::thread::id()) : producer(producer), consumer(consumer) {
		pFirst = pDivider = pLast = new Node(T());
	}

	~LFQueue1P1C() {
		while (pFirst != nullptr) {
			Node *pTemp = pFirst;
			pFirst = pTemp->pNext;
			delete pTemp;
		}
	}

	bool empty() {
		return !((*pDivider) != (*pLast));
	}

	int size() {
		Node *pCur = pDivider;
		int s = 0;
		while (pCur != pLast) {
			pCur = pCur->pNext;
			s++;
		}
		return s;
	}

	bool Consume(T& rValue, bool bRemoveOld = true) {
		if (consumer != std::this_thread::get_id())
			throw ConcurrencyError("Invalid consumer.");
		if ( (*pDivider) != (*pLast) ) {
			if (bRemoveOld) {
				while ((*pDivider).pNext != pLast) {
					pDivider.store((*pDivider).pNext);
				}
			}
			rValue = (*pDivider).pNext->mValue;
			pDivider.store((*pDivider).pNext);
			return true;
		}
		return false;
	}

	void Produce(const T &rNew) {
		if (producer != std::this_thread::get_id()) throw ConcurrencyError("Invalid producer.");
		(*pLast).pNext = new Node(rNew); // allocate a new node at (*pLast).pNext
		(*pLast).pNext->m_bValid = true;
		pLast.store((*pLast).pNext); // Update pLast pointing to the new node

		while (pFirst != pDivider) {
			Node *pTemp = pFirst;
			pFirst = pFirst->pNext;
			delete pTemp;
		}
	}

	void SetConsumer() {
		if (consumer != std::thread::id()) throw ConcurrencyError("Consumer already set");
		consumer = std::this_thread::get_id();
	}

	void SetProducer() {
		if (producer != std::thread::id()) throw ConcurrencyError("Producer already set");
		producer = std::this_thread::get_id();
	}
};
