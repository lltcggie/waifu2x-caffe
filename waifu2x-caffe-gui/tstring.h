#pragma once

#include <string>
#include <boost/filesystem.hpp>


#ifdef UNICODE
typedef std::wstring tstring;
inline tstring getTString(const boost::filesystem::path& p)
{
	return p.wstring();
}
#else
typedef std::string tstring;
inline tstring getTString(const boost::filesystem::path& p)
{
	return p.string();
}
#endif
