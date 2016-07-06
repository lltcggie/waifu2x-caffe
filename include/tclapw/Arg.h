// -*- Mode: c++; c-basic-offset: 4; tab-width: 4; -*-

/******************************************************************************
 *
 *  file:  Arg.h
 *
 *  Copyright (c) 2003, Michael E. Smoot .
 *  Copyright (c) 2004, Michael E. Smoot, Daniel Aarno .
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


#ifndef TCLAP_ARGUMENT_H
#define TCLAP_ARGUMENT_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#else
#define HAVE_SSTREAM
#endif

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <iomanip>
#include <cstdio>

#if defined(HAVE_SSTREAM)
#include <sstream>
typedef std::istringstream istringstream;
#elif defined(HAVE_STRSTREAM)
#include <strstream>
typedef std::istrstream istringstream;
#else
#error L"Need a stringstream (sstream or strstream) to compile!"
#endif

#include <tclapw/ArgException.h>
#include <tclapw/Visitor.h>
#include <tclapw/CmdLineInterface.h>
#include <tclapw/ArgTraits.h>
#include <tclapw/StandardTraits.h>

namespace TCLAP {

/**
 * A virtual base class that defines the essential data for all arguments.
 * This class, or one of its existing children, must be subclassed to do
 * anything.
 */
class Arg
{
	private:
		/**
		 * Prevent accidental copying.
		 */
		Arg(const Arg& rhs);

		/**
		 * Prevent accidental copying.
		 */
		Arg& operator=(const Arg& rhs);

		/**
		 * Indicates whether the rest of the arguments should be ignored.
		 */
		static bool& ignoreRestRef() { static bool ign = false; return ign; }

		/**
		 * The delimiter that separates an argument flag/name from the
		 * value.
		 */
		static wchar_t& delimiterRef() { static wchar_t delim = L' '; return delim; }

		static bool &ignoreMismatchedRef() { static bool ign = false; return ign; }

	protected:

		/**
		 * The single wchar_t flag used to identify the argument.
		 * This value (preceded by a dash {-}), can be used to identify
		 * an argument on the command line.  The _flag can be blank,
		 * in fact this is how unlabeled args work.  Unlabeled args must
		 * override appropriate functions to get correct handling. Note
		 * that the _flag does NOT include the dash as part of the flag.
		 */
		std::wstring _flag;

		/**
		 * A single work namd indentifying the argument.
		 * This value (preceded by two dashed {--}) can also be used
		 * to identify an argument on the command line.  Note that the
		 * _name does NOT include the two dashes as part of the _name. The
		 * _name cannot be blank.
		 */
		std::wstring _name;

		/**
		 * Description of the argument.
		 */
		std::wstring _description;

		/**
		 * Indicating whether the argument is required.
		 */
		bool _required;

		/**
		 * Label to be used in usage description.  Normally set to
		 * L"required", but can be changed when necessary.
		 */
		std::wstring _requireLabel;

		/**
		 * Indicates whether a value is required for the argument.
		 * Note that the value may be required but the argument/value
		 * combination may not be, as specified by _required.
		 */
		bool _valueRequired;

		/**
		 * Indicates whether the argument has been set.
		 * Indicates that a value on the command line has matched the
		 * name/flag of this argument and the values have been set accordingly.
		 */
		bool _alreadySet;

		/**
		 * A pointer to a vistitor object.
		 * The visitor allows special handling to occur as soon as the
		 * argument is matched.  This defaults to NULL and should not
		 * be used unless absolutely necessary.
		 */
		Visitor* _visitor;

		/**
		 * Whether this argument can be ignored, if desired.
		 */
		bool _ignoreable;

		/**
		 * Indicates that the arg was set as part of an XOR and not on the
		 * command line.
		 */
		bool _xorSet;

		bool _acceptsMultipleValues;

		/**
		 * Performs the special handling described by the Vistitor.
		 */
		void _checkWithVisitor() const;

		/**
		 * Primary constructor. YOU (yes you) should NEVER construct an Arg
		 * directly, this is a base class that is extended by various children
		 * that are meant to be used.  Use SwitchArg, ValueArg, MultiArg,
		 * UnlabeledValueArg, or UnlabeledMultiArg instead.
		 *
		 * \param flag - The flag identifying the argument.
		 * \param name - The name identifying the argument.
		 * \param desc - The description of the argument, used in the usage.
		 * \param req - Whether the argument is required.
		 * \param valreq - Whether the a value is required for the argument.
		 * \param v - The visitor checked by the argument. Defaults to NULL.
		 */
 		Arg( const std::wstring& flag,
			 const std::wstring& name,
			 const std::wstring& desc,
			 bool req,
			 bool valreq,
			 Visitor* v = NULL );

	public:
		/**
		 * Destructor.
		 */
		virtual ~Arg();

		/**
		 * Adds this to the specified list of Args.
		 * \param argList - The list to add this to.
		 */
		virtual void addToList( std::list<Arg*>& argList ) const;

		/**
		 * Begin ignoring arguments since the L"--" argument was specified.
		 */
		static void beginIgnoring() { ignoreRestRef() = true; }

		/**
		 * Whether to ignore the rest.
		 */
		static bool ignoreRest() { return ignoreRestRef(); }

		static void enableIgnoreMismatched() { ignoreMismatchedRef() = true; }
		static bool ignoreMismatched() { return ignoreMismatchedRef(); }

		/**
		 * The delimiter that separates an argument flag/name from the
		 * value.
		 */
		static wchar_t delimiter() { return delimiterRef(); }

		/**
		 * The wchar_t used as a place holder when SwitchArgs are combined.
		 * Currently set to the bell wchar_t (ASCII 7).
		 */
		static wchar_t blankChar() { return (wchar_t)7; }

		/**
		 * The wchar_t that indicates the beginning of a flag.  Defaults to L'-', but
		 * clients can define TCLAP_FLAGSTARTCHAR to override.
		 */
#ifndef TCLAP_FLAGSTARTCHAR
#define TCLAP_FLAGSTARTCHAR L'-'
#endif
		static wchar_t flagStartChar() { return TCLAP_FLAGSTARTCHAR; }

		/**
		 * The sting that indicates the beginning of a flag.  Defaults to L"-", but
		 * clients can define TCLAP_FLAGSTARTSTRING to override. Should be the same
		 * as TCLAP_FLAGSTARTCHAR.
		 */
#ifndef TCLAP_FLAGSTARTSTRING
#define TCLAP_FLAGSTARTSTRING L"-"
#endif
		static const std::wstring flagStartString() { return TCLAP_FLAGSTARTSTRING; }

		/**
		 * The sting that indicates the beginning of a name.  Defaults to L"--", but
		 *  clients can define TCLAP_NAMESTARTSTRING to override.
		 */
#ifndef TCLAP_NAMESTARTSTRING
#define TCLAP_NAMESTARTSTRING L"--"
#endif
		static const std::wstring nameStartString() { return TCLAP_NAMESTARTSTRING; }

		/**
		 * The name used to identify the ignore rest argument.
		 */
		static const std::wstring ignoreNameString() { return L"ignore_rest"; }

		/**
		 * Sets the delimiter for all arguments.
		 * \param c - The character that delimits flags/names from values.
		 */
		static void setDelimiter( wchar_t c ) { delimiterRef() = c; }

		/**
		 * Pure virtual method meant to handle the parsing and value assignment
		 * of the string on the command line.
		 * \param i - Pointer the the current argument in the list.
		 * \param args - Mutable list of strings. What is
		 * passed in from main.
		 */
		virtual bool processArg(int *i, std::vector<std::wstring>& args) = 0;

		/**
		 * Operator ==.
		 * Equality operator. Must be virtual to handle unlabeled args.
		 * \param a - The Arg to be compared to this.
		 */
		virtual bool operator==(const Arg& a) const;

		/**
		 * Returns the argument flag.
		 */
		const std::wstring& getFlag() const;

		/**
		 * Returns the argument name.
		 */
		const std::wstring& getName() const;

		/**
		 * Returns the argument description.
		 */
		std::wstring getDescription() const;

		/**
		 * Indicates whether the argument is required.
		 */
		virtual bool isRequired() const;

		/**
		 * Sets _required to true. This is used by the XorHandler.
		 * You really have no reason to ever use it.
		 */
		void forceRequired();

		/**
		 * Sets the _alreadySet value to true.  This is used by the XorHandler.
		 * You really have no reason to ever use it.
		 */
		void xorSet();

		/**
		 * Indicates whether a value must be specified for argument.
		 */
		bool isValueRequired() const;

		/**
		 * Indicates whether the argument has already been set.  Only true
		 * if the arg has been matched on the command line.
		 */
		bool isSet() const;

		/**
		 * Indicates whether the argument can be ignored, if desired.
		 */
		bool isIgnoreable() const;

		/**
		 * A method that tests whether a string matches this argument.
		 * This is generally called by the processArg() method.  This
		 * method could be re-implemented by a child to change how
		 * arguments are specified on the command line.
		 * \param s - The string to be compared to the flag/name to determine
		 * whether the arg matches.
		 */
		virtual bool argMatches( const std::wstring& s ) const;

		/**
		 * Returns a simple string representation of the argument.
		 * Primarily for debugging.
		 */
		virtual std::wstring toString() const;

		/**
		 * Returns a short ID for the usage.
		 * \param valueId - The value used in the id.
		 */
		virtual std::wstring shortID( const std::wstring& valueId = L"val" ) const;

		/**
		 * Returns a long ID for the usage.
		 * \param valueId - The value used in the id.
		 */
		virtual std::wstring longID( const std::wstring& valueId = L"val" ) const;

		/**
		 * Trims a value off of the flag.
		 * \param flag - The string from which the flag and value will be
		 * trimmed. Contains the flag once the value has been trimmed.
		 * \param value - Where the value trimmed from the string will
		 * be stored.
		 */
		virtual void trimFlag( std::wstring& flag, std::wstring& value ) const;

		/**
		 * Checks whether a given string has blank chars, indicating that
		 * it is a combined SwitchArg.  If so, return true, otherwise return
		 * false.
		 * \param s - string to be checked.
		 */
		bool _hasBlanks( const std::wstring& s ) const;

		/**
		 * Sets the requireLabel. Used by XorHandler.  You shouldnL't ever
		 * use this.
		 * \param s - Set the requireLabel to this value.
		 */
		void setRequireLabel( const std::wstring& s );

		/**
		 * Used for MultiArgs and XorHandler to determine whether args
		 * can still be set.
		 */
		virtual bool allowMore();

		/**
		 * Use by output classes to determine whether an Arg accepts
		 * multiple values.
		 */
		virtual bool acceptsMultipleValues();

		/**
		 * Clears the Arg object and allows it to be reused by new
		 * command lines.
		 */
		 virtual void reset();
};

/**
 * Typedef of an Arg list iterator.
 */
typedef std::list<Arg*>::iterator ArgListIterator;

/**
 * Typedef of an Arg vector iterator.
 */
typedef std::vector<Arg*>::iterator ArgVectorIterator;

/**
 * Typedef of a Visitor list iterator.
 */
typedef std::list<Visitor*>::iterator VisitorListIterator;

/*
 * Extract a value of type T from it's string representation contained
 * in strVal. The ValueLike parameter used to select the correct
 * specialization of ExtractValue depending on the value traits of T.
 * ValueLike traits use operator>> to assign the value from strVal.
 */
template<typename T> void
ExtractValue(T &destVal, const std::wstring& strVal, ValueLike vl)
{
    static_cast<void>(vl); // Avoid warning about unused vl
    std::wistringstream is(strVal);

    int valuesRead = 0;
    while ( is.good() ) {
	if ( is.peek() != EOF )
#ifdef TCLAP_SETBASE_ZERO
	    is >> std::setbase(0) >> destVal;
#else
	    is >> destVal;
#endif
	else
	    break;

	valuesRead++;
    }

    if ( is.fail() )
	throw( ArgParseException(L"Couldn't read argument value "
				 L"from string '" + strVal + L"'"));


    if ( valuesRead > 1 )
	throw( ArgParseException(L"More than one valid value parsed from "
				 L"string '" + strVal + L"'"));

}

/*
 * Extract a value of type T from itL's string representation contained
 * in strVal. The ValueLike parameter used to select the correct
 * specialization of ExtractValue depending on the value traits of T.
 * StringLike uses assignment (operator=) to assign from strVal.
 */
template<typename T> void
ExtractValue(T &destVal, const std::wstring& strVal, StringLike sl)
{
    static_cast<void>(sl); // Avoid warning about unused sl
    SetString(destVal, strVal);
}

//////////////////////////////////////////////////////////////////////
//BEGIN Arg.cpp
//////////////////////////////////////////////////////////////////////

inline Arg::Arg(const std::wstring& flag,
         const std::wstring& name,
         const std::wstring& desc,
         bool req,
         bool valreq,
         Visitor* v) :
  _flag(flag),
  _name(name),
  _description(desc),
  _required(req),
  _requireLabel(L"required"),
  _valueRequired(valreq),
  _alreadySet(false),
  _visitor( v ),
  _ignoreable(true),
  _xorSet(false),
  _acceptsMultipleValues(false)
{
	if ( _flag.length() > 1 )
		throw(SpecificationException(
				L"Argument flag can only be one character long", toString() ) );

	if ( _name != ignoreNameString() &&
		 ( _flag == Arg::flagStartString() ||
		   _flag == Arg::nameStartString() ||
		   _flag == L" " ) )
		throw(SpecificationException(L"Argument flag cannot be either '" +
							Arg::flagStartString() + L"' or '" +
							Arg::nameStartString() + L"' or a space.",
							toString() ) );

	if ( ( _name.substr( 0, Arg::flagStartString().length() ) == Arg::flagStartString() ) ||
		 ( _name.substr( 0, Arg::nameStartString().length() ) == Arg::nameStartString() ) ||
		 ( _name.find(L" ", 0 ) != std::wstring::npos ) )
		throw(SpecificationException(L"Argument name begin with either '" +
							Arg::flagStartString() + L"' or '" +
							Arg::nameStartString() + L"' or space.",
							toString() ) );

}

inline Arg::~Arg() { }

inline std::wstring Arg::shortID( const std::wstring& valueId ) const
{
	std::wstring id = L"";

	if ( _flag != L"" )
		id = Arg::flagStartString() + _flag;
	else
		id = Arg::nameStartString() + _name;

	if ( _valueRequired )
		id += std::wstring( 1, Arg::delimiter() ) + L"<" + valueId  + L">";

	if ( !_required )
		id = L"[" + id + L"]";

	return id;
}

inline std::wstring Arg::longID( const std::wstring& valueId ) const
{
	std::wstring id = L"";

	if ( _flag != L"" )
	{
		id += Arg::flagStartString() + _flag;

		if ( _valueRequired )
			id += std::wstring( 1, Arg::delimiter() ) + L"<" + valueId + L">";

		id += L",  ";
	}

	id += Arg::nameStartString() + _name;

	if ( _valueRequired )
		id += std::wstring( 1, Arg::delimiter() ) + L"<" + valueId + L">";

	return id;

}

inline bool Arg::operator==(const Arg& a) const
{
	if ( ( _flag != L"" && _flag == a._flag ) || _name == a._name)
		return true;
	else
		return false;
}

inline std::wstring Arg::getDescription() const
{
	std::wstring desc = L"";
	if ( _required )
		desc = L"(" + _requireLabel + L")  ";

//	if ( _valueRequired )
//		desc += "(value required)  L";

	desc += _description;
	return desc;
}

inline const std::wstring& Arg::getFlag() const { return _flag; }

inline const std::wstring& Arg::getName() const { return _name; }

inline bool Arg::isRequired() const { return _required; }

inline bool Arg::isValueRequired() const { return _valueRequired; }

inline bool Arg::isSet() const
{
	if ( _alreadySet && !_xorSet )
		return true;
	else
		return false;
}

inline bool Arg::isIgnoreable() const { return _ignoreable; }

inline void Arg::setRequireLabel( const std::wstring& s)
{
	_requireLabel = s;
}

inline bool Arg::argMatches( const std::wstring& argFlag ) const
{
	if ( ( argFlag == Arg::flagStartString() + _flag && _flag != L"" ) ||
	       argFlag == Arg::nameStartString() + _name )
		return true;
	else
		return false;
}

inline std::wstring Arg::toString() const
{
	std::wstring s = L"";

	if ( _flag != L"" )
		s += Arg::flagStartString() + _flag + L" ";

	s += L"(" + Arg::nameStartString() + _name + L")";

	return s;
}

inline void Arg::_checkWithVisitor() const
{
	if ( _visitor != NULL )
		_visitor->visit();
}

/**
 * Implementation of trimFlag.
 */
inline void Arg::trimFlag(std::wstring& flag, std::wstring& value) const
{
	int stop = 0;
	for ( int i = 0; static_cast<unsigned int>(i) < flag.length(); i++ )
		if ( flag[i] == Arg::delimiter() )
		{
			stop = i;
			break;
		}

	if ( stop > 1 )
	{
		value = flag.substr(stop+1);
		flag = flag.substr(0,stop);
	}

}

/**
 * Implementation of _hasBlanks.
 */
inline bool Arg::_hasBlanks( const std::wstring& s ) const
{
	for ( int i = 1; static_cast<unsigned int>(i) < s.length(); i++ )
		if ( s[i] == Arg::blankChar() )
			return true;

	return false;
}

inline void Arg::forceRequired()
{
	_required = true;
}

inline void Arg::xorSet()
{
	_alreadySet = true;
	_xorSet = true;
}

/**
 * Overridden by Args that need to added to the end of the list.
 */
inline void Arg::addToList( std::list<Arg*>& argList ) const
{
	argList.push_front( const_cast<Arg*>(this) );
}

inline bool Arg::allowMore()
{
	return false;
}

inline bool Arg::acceptsMultipleValues()
{
	return _acceptsMultipleValues;
}

inline void Arg::reset()
{
	_xorSet = false;
	_alreadySet = false;
}

//////////////////////////////////////////////////////////////////////
//END Arg.cpp
//////////////////////////////////////////////////////////////////////

} //namespace TCLAP

#endif

