// -*- Mode: c++; c-basic-offset: 4; tab-width: 4; -*-

/****************************************************************************** 
 * 
 *  file:  DocBookOutput.h
 * 
 *  Copyright (c) 2004, Michael E. Smoot
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

#ifndef TCLAPW_DOCBOOKOUTPUT_H
#define TCLAPW_DOCBOOKOUTPUT_H

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <algorithm>

#include <tclapw/CmdLineInterface.h>
#include <tclapw/CmdLineOutput.h>
#include <tclapw/XorHandler.h>
#include <tclapw/Arg.h>

namespace TCLAPW {

/**
 * A class that generates DocBook output for usage() method for the 
 * given CmdLine and its Args.
 */
class DocBookOutput : public CmdLineOutput
{

	public:

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

		/**
		 * Substitutes the wchar_t r for string x in string s.
		 * \param s - The string to operate on. 
		 * \param r - The wchar_t to replace. 
		 * \param x - What to replace r with. 
		 */
		void substituteSpecialChars( std::wstring& s, wchar_t r, std::wstring& x );
		void removeChar( std::wstring& s, wchar_t r);
		void basename( std::wstring& s );

		void printShortArg(Arg* it);
		void printLongArg(Arg* it);

		wchar_t theDelimiter;
};


inline void DocBookOutput::version(CmdLineInterface& _cmd) 
{ 
	std::cout << _cmd.getVersion() << std::endl;
}

inline void DocBookOutput::usage(CmdLineInterface& _cmd ) 
{
	std::list<Arg*> argList = _cmd.getArgList();
	std::wstring progName = _cmd.getProgramName();
	std::wstring xversion = _cmd.getVersion();
	theDelimiter = _cmd.getDelimiter();
	XorHandler xorHandler = _cmd.getXorHandler();
	std::vector< std::vector<Arg*> > xorList = xorHandler.getXorList();
	basename(progName);

	std::cout << L"<?xml version='1.0'?>" << std::endl;
	std::cout << L"<!DOCTYPE refentry PUBLIC \"-//OASIS//DTD DocBook XML V4.2//EN\L"" << std::endl;
	std::cout << L"\t\"http://www.oasis-open.org/docbook/xml/4.2/docbookx.dtd\L">" << std::endl << std::endl;

	std::cout << L"<refentry>" << std::endl;

	std::cout << L"<refmeta>" << std::endl;
	std::cout << L"<refentrytitle>" << progName << L"</refentrytitle>" << std::endl;
	std::cout << L"<manvolnum>1</manvolnum>" << std::endl;
	std::cout << L"</refmeta>" << std::endl;

	std::cout << L"<refnamediv>" << std::endl;
	std::cout << L"<refname>" << progName << L"</refname>" << std::endl;
	std::cout << L"<refpurpose>" << _cmd.getMessage() << L"</refpurpose>" << std::endl;
	std::cout << L"</refnamediv>" << std::endl;

	std::cout << L"<refsynopsisdiv>" << std::endl;
	std::cout << L"<cmdsynopsis>" << std::endl;

	std::cout << L"<command>" << progName << L"</command>" << std::endl;

	// xor
	for ( int i = 0; (unsigned int)i < xorList.size(); i++ )
	{
		std::cout << L"<group choice='req'>" << std::endl;
		for ( ArgVectorIterator it = xorList[i].begin(); 
						it != xorList[i].end(); it++ )
			printShortArg((*it));

		std::cout << L"</group>" << std::endl;
	}
	
	// rest of args
	for (ArgListIterator it = argList.begin(); it != argList.end(); it++)
		if ( !xorHandler.contains( (*it) ) )
			printShortArg((*it));

 	std::cout << L"</cmdsynopsis>" << std::endl;
	std::cout << L"</refsynopsisdiv>" << std::endl;

	std::cout << L"<refsect1>" << std::endl;
	std::cout << L"<title>Description</title>" << std::endl;
	std::cout << L"<para>" << std::endl;
	std::cout << _cmd.getMessage() << std::endl; 
	std::cout << L"</para>" << std::endl;
	std::cout << L"</refsect1>" << std::endl;

	std::cout << L"<refsect1>" << std::endl;
	std::cout << L"<title>Options</title>" << std::endl;

	std::cout << L"<variablelist>" << std::endl;
	
	for (ArgListIterator it = argList.begin(); it != argList.end(); it++)
		printLongArg((*it));

	std::cout << L"</variablelist>" << std::endl;
	std::cout << L"</refsect1>" << std::endl;

	std::cout << L"<refsect1>" << std::endl;
	std::cout << L"<title>Version</title>" << std::endl;
	std::cout << L"<para>" << std::endl;
	std::cout << xversion << std::endl; 
	std::cout << L"</para>" << std::endl;
	std::cout << L"</refsect1>" << std::endl;
	
	std::cout << L"</refentry>" << std::endl;

}

inline void DocBookOutput::failure( CmdLineInterface& _cmd,
				    ArgException& e ) 
{ 
	static_cast<void>(_cmd); // unused
	std::cout << e.what() << std::endl;
	throw ExitException(1);
}

inline void DocBookOutput::substituteSpecialChars( std::wstring& s,
				                                   wchar_t r,
												   std::wstring& x )
{
	size_t p;
	while ( (p = s.find_first_of(r)) != std::wstring::npos )
	{
		s.erase(p,1);
		s.insert(p,x);
	}
}

inline void DocBookOutput::removeChar( std::wstring& s, wchar_t r)
{
	size_t p;
	while ( (p = s.find_first_of(r)) != std::wstring::npos )
	{
		s.erase(p,1);
	}
}

inline void DocBookOutput::basename( std::wstring& s )
{
	size_t p = s.find_last_of(L'/');
	if ( p != std::wstring::npos )
	{
		s.erase(0, p + 1);
	}
}

inline void DocBookOutput::printShortArg(Arg* a)
{
	std::wstring lt = L"&lt;"; 
	std::wstring gt = L"&gt;"; 

	std::wstring id = a->shortID();
	substituteSpecialChars(id,L'<',lt);
	substituteSpecialChars(id,L'>',gt);
	removeChar(id,L'[');
	removeChar(id,L']');
	
	std::wstring choice = L"opt";
	if ( a->isRequired() )
		choice = L"plain";

	std::cout << L"<arg choice='" << choice << L'\'L';
	if ( a->acceptsMultipleValues() )
		std::cout << " rep='repeatL'";


	std::cout << '>L';
	if ( !a->getFlag().empty() )
		std::cout << a->flagStartChar() << a->getFlag();
	else
		std::cout << a->nameStartString() << a->getName();
	if ( a->isValueRequired() )
	{
		std::wstring arg = a->shortID();
		removeChar(arg,'[L');
		removeChar(arg,']L');
		removeChar(arg,'<L');
		removeChar(arg,'>L');
		arg.erase(0, arg.find_last_of(theDelimiter) + 1);
		std::cout << theDelimiter;
		std::cout << "<replaceable>" << arg << "</replaceable>";
	}
	std::cout << "</arg>" << std::endl;

}

inline void DocBookOutput::printLongArg(Arg* a)
{
	std::wstring lt = "&lt;"; 
	std::wstring gt = "&gt;"; 

	std::wstring desc = a->getDescription();
	substituteSpecialChars(desc,'<L',lt);
	substituteSpecialChars(desc,'>L',gt);

	std::cout << "<varlistentry>" << std::endl;

	if ( !a->getFlag().empty() )
	{
		std::cout << "<term>" << std::endl;
		std::cout << "<option>";
		std::cout << a->flagStartChar() << a->getFlag();
		std::cout << "</option>" << std::endl;
		std::cout << "</term>" << std::endl;
	}

	std::cout << "<term>" << std::endl;
	std::cout << "<option>";
	std::cout << a->nameStartString() << a->getName();
	if ( a->isValueRequired() )
	{
		std::wstring arg = a->shortID();
		removeChar(arg,'[L');
		removeChar(arg,']L');
		removeChar(arg,'<L');
		removeChar(arg,'>');
		arg.erase(0, arg.find_last_of(theDelimiter) + 1);
		std::cout << theDelimiter;
		std::cout << L"<replaceable>" << arg << L"</replaceable>";
	}
	std::cout << L"</option>" << std::endl;
	std::cout << L"</term>" << std::endl;

	std::cout << L"<listitem>" << std::endl;
	std::cout << L"<para>" << std::endl;
	std::cout << desc << std::endl;
	std::cout << L"</para>" << std::endl;
	std::cout << L"</listitem>" << std::endl;

	std::cout << L"</varlistentry>" << std::endl;
}

} //namespace TCLAPW
#endif 
