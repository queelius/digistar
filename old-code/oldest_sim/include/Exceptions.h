#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>
#include <string>

const int EXCEPTION_INVALID_ARG             = 1;
const int EXCEPTION_INVALID_NUMBER_FORMAT   = 2;
const int EXCEPTION_INVALID_DATE_FORMAT     = 3;
const int EXCEPTION_NOT_IMPLEMENTED         = 4;
const int EXCEPTION_UNSUPPORTED_OPERATION   = 5;
const int EXCEPTION_ITERATOR_OUT_OF_RANGE   = 6;
const int EXCEPTION_CYCLE_FOUND             = 7;
const int EXCEPTION_PATH_NOT_FOUND          = 8;
const int EXCEPTION_NODE_NOT_FOUND          = 9;
const int EXCEPTION_NODE_ALREADY_EXISTS     = 10;
const int EXCEPTION_MEMORY_NOT_ALLOCATED    = 11;
const int EXCEPTION_EDGE_NOT_FOUND          = 12;
const int EXCEPTION_EDGE_ALREADY_EXISTS     = 13;
const int EXCEPTION_INDEX_NOT_VALID         = 14;

class Exception: public std::exception
{
public:
    Exception(const char* msg):
      std::exception(msg), errorCode(0) {};

    Exception(const std::string& msg):
      std::exception(msg.c_str()), errorCode(0) {};

    Exception(unsigned int errorCode):
      std::exception(), errorCode(errorCode) {};

    Exception(const char* msg, unsigned int errorCode):
      std::exception(msg), errorCode(0) {};

    Exception(const std::string& msg, unsigned int errorCode):
      std::exception(msg.c_str()), errorCode(0) {};

    virtual std::string getMessage() const { return std::string(what()); };
    virtual int getErrorCode() const { return errorCode; };

protected:
    unsigned int errorCode;
};

class InvalidArgument: public Exception
{
public: InvalidArgument(const char* msg = "Invalid Argument(s)"):
    Exception(msg, EXCEPTION_INVALID_ARG) {};
};

class InvalidNumberFormat: public Exception
{
public: InvalidNumberFormat(const char* msg = "Invalid Number Format"):
    Exception(msg, EXCEPTION_INVALID_NUMBER_FORMAT) {};
};

class InvalidDateFormat: public Exception
{
public: InvalidDateFormat(const char* msg = "Invalid Date Format"):
    Exception(msg, EXCEPTION_INVALID_DATE_FORMAT) {};
};

class NotImplemented: public Exception
{
public: NotImplemented(const char* msg = "Not Implemented"):
    Exception(msg, EXCEPTION_NOT_IMPLEMENTED) {};
};

class UnsupportedOperation: public Exception
{
public: UnsupportedOperation(const char* msg = "Unsupported Operation"):
    Exception(msg, EXCEPTION_UNSUPPORTED_OPERATION) {};
};

class IteratorOutOfRange: public Exception
{
public: IteratorOutOfRange(const char* msg = "Iterator Out Of Range"):
    Exception(msg, EXCEPTION_ITERATOR_OUT_OF_RANGE) {};
};

class CycleFound: public Exception
{
public: CycleFound(const char* msg = "Cycle Found"):
    Exception(msg, EXCEPTION_CYCLE_FOUND) {};
};

class PathNotFound: public Exception
{
public: PathNotFound(const char* msg = "Path Not Found"):
    Exception(msg, EXCEPTION_PATH_NOT_FOUND) {};
};

class NodeNotFound: public Exception
{
public: NodeNotFound(const char* msg = "Node Not Found"):
    Exception(msg, EXCEPTION_NODE_NOT_FOUND) {};
};

class NodeAlreadyExists: public Exception
{
public: NodeAlreadyExists(const char* msg = "Node Already Exists"):
    Exception(msg, EXCEPTION_NODE_ALREADY_EXISTS) {};
};

class EdgeAlreadyExists: public Exception
{
public: EdgeAlreadyExists(const char* msg = "Edge Already Exists"):
    Exception(msg, EXCEPTION_EDGE_ALREADY_EXISTS) {};
};

class EdgeNotFound: public Exception
{
public: EdgeNotFound(const char* msg = "Edge Not Found"):
    Exception(msg, EXCEPTION_EDGE_NOT_FOUND) {};
};

class IndexNotValid: public Exception
{
public: IndexNotValid(const char* msg = "Index Not Valid"):
    Exception(msg, EXCEPTION_INDEX_NOT_VALID) {};

protected:
};

class MemoryNotAllocated: public Exception
{
public: MemoryNotAllocated(const char* msg = "Memory Not Allocated"):
    Exception(msg) {};
};

#endif