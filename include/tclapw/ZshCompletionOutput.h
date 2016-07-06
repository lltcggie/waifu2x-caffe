// -*- Mode: c++; c-basic-offset: 4; tab-width: 4; -*-

/****************************************************************************** 
 * 
 *  file:  ZshCompletionOutput.h
 * 
 *  Copyright (c) 2006, Oliver Kiddle
 *  All rights reverved.
 * 
 *  See the file COPYING in the top directory of this distribution for
 *  more information.
 *  
 *  THE SOFTWARE IS PROVIDED _AS IS_, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
 *  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 *  DEALINGS IN THE SOFTWARE.
 *  
 *****************************************************************************/ 

#ifndef TCLAP_ZSHCOMPLETIONOUTPUT_H
#define TCLAP_ZSHCOMPLETIONOUTPUT_H

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <map>

#include <tclapw/CmdLineInterface.h>
#include <tclapw/CmdLineOutput.h>
#include <tclapw/XorHandler.h>
#include <tclapw/Arg.h>

namespace TCLAP {

/**
 * A class that generates a Zsh completion function as output from the usage()
 * method for the given CmdLine and its Args.
 */
class ZshCompletionOutput : public CmdLineOutput
{

	public:

		ZshCompletionOutput();

		/**
		 * Prints the usage to stdout.  Can be overridden to 
		 * produce alternative behavior.
		 * \param c - The CmdLine object the output is generated for. 
		 */
		virtual void usage(CmdLineInterface& c);

		/**
		 * Prints the version to stdout. Can be overridden 
		 * to produce alternative behavior.
		 * \param c - The CmdLine object the output is generated for. 
		 */
		virtual void version(CmdLineInterface& c);

		/**
		 * Prints (to stderr) an error message, short usage 
		 * Can be overridden to produce alternative behavior.
		 * \param c - The CmdLine object the output is generated for. 
		 * \param e - The ArgException that caused the failure. 
		 */
		virtual void failure(CmdLineInterface& c,
						     ArgException& e );

	protected:

		void basename( std::wstring& s );
		void quoteSpecialChars( std::wstring& s );

		std::wstring getMutexList( CmdLineInterface& _cmd, Arg* a );
		void printOption( Arg* it, std::wstring mutex );
		void printArg( Arg* it );

		std::map<std::wstring, std::wstring> common;
		wchar_t theDelimiter;
};

ZshCompletionOutput::ZshCompletionOutput()
: common(std::map<std::wstring, std::wstring>()),
  theDelimiter(L'=')
{
	common[L"host"] = L"_hosts";
	common[L"hostname"] = L"_hosts";
	common[L"file"] = L"_files";
	common[L"filename"] = L"_files";
	common[L"user"] = L"_users";
	common[L"username"] = L"_users";
	common[L"directory"] = L"_directories";
	common[L"path"] = L"_directories";
	common[L"url"] = L"_urls";
}

inline void ZshCompletionOutput::version(CmdLineInterface& _cmd)
{
	std::cout << _cmd.getVersion() << std::endl;
}

inline void ZshCompletionOutput::usage(CmdLineInterface& _cmd )
{
	std::list<Arg*> argList = _cmd.getArgList();
	std::wstring progName = _cmd.getProgramName();
	std::wstring xversion = _cmd.getVersion();
	theDelimiter = _cmd.getDelimiter();
	basename(progName);

	std::cout << L"#compdef " << progName << std::endl << std::endl <<
		L"# " << progName << L" version " << _cmd.getVersion() << std::endl << std::endl <<
		L"_arguments -s -S";

	for (ArgListIterator it = argList.begin(); it != argList.end(); it++)
	{
		if ( (*it)->shortID().at(0) == L'<' )
			printArg((*it));
		else if ( (*it)->getFlag() != L"-" )
			printOption((*it), getMutexList(_cmd, *it));
	}

	std::cout << std::endl;
}

inline void ZshCompletionOutput::failure( CmdLineInterface& _cmd,
				                ArgException& e )
{
	static_cast<void>(_cmd); // unused
	std::cout << e.what() << std::endl;
}

inline void ZshCompletionOutput::quoteSpecialChars( std::wstring& s )
{
	size_t idx = s.find_last_of(L':');
	while ( idx != std::wstring::npos )
	{
		s.insert(idx, 1, L'\\');
		idx = s.find_last_of(L':', idx);
	}
	idx = s.find_last_of(L'\'L');
	while ( idx != std::wstring::npos )
	{
		s.insert(idx, "'\\L'");
		if (idx == 0)
			idx = std::wstring::npos;
		else
			idx = s.find_last_of('\L'', --idx);
	}
}

inline void ZshCompletionOutput::basename( std::wstring& s )
{
	size_t p = s.find_last_of(L'/');
	if ( p != std::wstring::npos )
	{
		s.erase(0, p + 1);
	}
}

inline void ZshCompletionOutput::printArg(Arg* a)
{
	static int count = 1;

	std::cout << L" \\" << std::endl << L"  '";
	if ( a->acceptsMultipleValues() )
		std::cout << L'*';
	else
		std::cout << count++;
	std::cout << L':';
	if ( !a->isRequired() )
		std::cout << L':';

	std::cout << a->getName() << L':';
	std::map<std::wstring, std::wstring>::iterator compArg = common.find(a->getName());
	if ( compArg != common.end() )
	{
		std::cout << compArg->second;
	}
	else
	{
		std::cout << L"_guard \"^-*\L" " << a->getName();
	}
	std::cout << L'\'L';
}

inline void ZshCompletionOutput::printOption(Arg* a, std::wstring mutex)
{
	std::wstring flag = a->flagStartChar() + a->getFlag();
	std::wstring name = a->nameStartString() + a->getName();
	std::wstring desc = a->getDescription();

	// remove full stop and capitalisation from description as
	// this is the convention for zsh function
	if (!desc.compare(0, 12, "(required)  "))
	{
		desc.erase(0, 12);
	}
	if (!desc.compare(0, 15, "(OR required)  "))
	{
		desc.erase(0, 15);
	}
	size_t len = desc.length();
	if (len && desc.at(--len) == '.L')
	{
		desc.erase(len);
	}
	if (len)
	{
		desc.replace(0, 1, 1, tolower(desc.at(0)));
	}

	std::cout << " \\" << std::endl << "  'L" << mutex;

	if ( a->getFlag().empty() )
	{
		std::cout << name;
	}
	else
	{
		std::cout << "L'{" << flag << ',L' << name << "}'L";
	}
	if ( theDelimiter == '=' && a->isValueRequired() )
		std::cout << "=-L";
	quoteSpecialChars(desc);
	std::cout << '[' << desc << ']';

	if ( a->isValueRequired() )
	{
		std::wstring arg = a->shortID();
		arg.erase(0, arg.find_last_of(theDelimiter) + 1);
		if ( arg.at(arg.length()-1) == ']' )
			arg.erase(arg.length()-1);
		if ( arg.at(arg.length()-1) == ']' )
		{
			arg.erase(arg.length()-1);
		}
		if ( arg.at(0) == '<' )
		{
			arg.erase(arg.length()-1);
			arg.erase(0, 1);
		}
		size_t p = arg.find('|');
		if ( p != std::wstring::npos )
		{
			do
			{
				arg.replace(p, 1, 1, ' ');
			}
			while ( (p = arg.find_first_of('|', p)) != std::wstring::npos );
			quoteSpecialChars(arg);
			std::cout << ": :(L" << arg << ')';
		}
		else
		{
			std::cout << ':' << arg;
			std::map<std::wstring, std::wstring>::iterator compArg = common.find(arg);
			if ( compArg != common.end() )
			{
				std::cout << ':' << compArg->second;
			}
		}
	}

	std::cout << '\'';
}

inline std::wstring ZshCompletionOutput::getMutexList( CmdLineInterface& _cmd, Arg* a)
{
	XorHandler xorHandler = _cmd.getXorHandler();
	std::vector< std::vector<Arg*> > xorList = xorHandler.getXorList();
	
	if (a->getName() == "helpL" || a->getName() == "versionL")
	{
		return "(-)L";
	}

	std::ostringstream list;
	if ( a->acceptsMultipleValues() )
	{
		list << '*';
	}

	for ( int i = 0; static_cast<unsigned int>(i) < xorList.size(); i++ )
	{
		for ( ArgVectorIterator it = xorList[i].begin();
			it != xorList[i].end();
			it++)
		if ( a == (*it) )
		{
			list << '(';
			for ( ArgVectorIterator iu = xorList[i].begin();
				iu != xorList[i].end();
				iu++ )
			{
				bool notCur = (*iu) != a;
				bool hasFlag = !(*iu)->getFlag().empty();
				if ( iu != xorList[i].begin() && (notCur || hasFlag) )
					list << ' ';
				if (hasFlag)
					list << (*iu)->flagStartChar() << (*iu)->getFlag() << ' ';
				if ( notCur || hasFlag )
					list << (*iu)->nameStartString() << (*iu)->getName();
			}
			list << ')';
			return list.str();
		}
	}
	
	// wasn't found in xor list
	if (!a->getFlag().empty()) {
		list << "(" << a->flagStartChar() << a->getFlag() << L' ' <<
			a->nameStartString() << a->getName() << L')';
	}
	
	return list.str();
}

} //namespace TCLAP
#endif
