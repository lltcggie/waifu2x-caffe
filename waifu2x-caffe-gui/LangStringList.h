#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <vector>
#include <string>
#include <boost/tokenizer.hpp>
#include <unordered_map>
#include <rapidjson/document.h>
#include <stdio.h>
#include "tstring.h"


class LangStringList
{
public:
	struct stLangSetting
	{
		std::wstring LangName;
		WORD LangID;
		WORD SubLangID;
		std::wstring FileName;

		stLangSetting() : LangID(0), SubLangID(0)
		{}
	};

private:
	tstring LangBaseDir;
	std::vector<stLangSetting> LangList;
	stLangSetting NowLang;
	LANGID NowLangID = GetUserDefaultUILanguage();
	std::unordered_map<std::wstring, std::wstring> LangStringMap;
	std::unordered_map<std::wstring, std::wstring> DefaultLangStringMap; // 現在の言語になかった文字列のフォールバック用。リストの一番最初に書かれている言語がデフォルト

private:
	static std::wstring Utf8ToUtf16(const char *src, int src_size = -1)
	{
		int ret;

		ret = MultiByteToWideChar(CP_UTF8, 0, src, src_size, NULL, 0);
		if (ret == 0)
			return std::wstring();

		std::vector<wchar_t> buf(ret);

		ret = MultiByteToWideChar(CP_UTF8, 0, src, src_size, buf.data(), ret);
		if (ret == 0)
			return std::wstring();

		if (buf.back() != L'\0')
			buf.push_back(L'\0');

		return std::wstring(buf.data());
	}

	static int getNum(const std::string &str)
	{
		if (str.length() >= 3 && str.substr(0, 2) == "0x")
			return strtol(str.c_str(), NULL, 16);
		else
			return strtol(str.c_str(), NULL, 10);
	}

	const stLangSetting& GetLang(const LANGID id) const
	{
		const auto Primarylang = PRIMARYLANGID(id);
		const auto Sublang = SUBLANGID(id);

		int FindPrimary = -1;
		int FindSub = -1;

		for (size_t i = 0; i < LangList.size(); i++)
		{
			const auto &l = LangList[i];

			if (Primarylang == l.LangID)
			{
				FindPrimary = (int)i;
				if (Primarylang == l.SubLangID)
				{
					FindSub = (int)i;
					break;
				}
			}
		}

		if (FindPrimary >= 0 && FindSub >= 0) // 現在の言語にピッタリ合うやつが見つかった
			return LangList[FindSub];
		else if (FindPrimary >= 0) // 現在の言語に属するものが見つかった
			return LangList[FindPrimary];

		// 見つからなかったから一番最初に書かれているやつにする
		if (LangList.size() > 0)
			return LangList[0];

		return stLangSetting();
	}

	void ReadLangFile(const stLangSetting &lang, std::unordered_map<std::wstring, std::wstring> &langStringMap) const
	{
		langStringMap.clear();

		if (lang.FileName.length() == 0)
			return;

		const auto LangFilePath = LangBaseDir + lang.FileName;

		rapidjson::Document d;
		std::vector<char> jsonBuf;

		FILE *fp = nullptr;
		try
		{
			fp = _wfopen(LangFilePath.c_str(), L"rb");
			if (!fp)
				return;

			fseek(fp, 0, SEEK_END);
			const auto size = ftell(fp);
			fseek(fp, 0, SEEK_SET);

			jsonBuf.resize(size + 1);
			fread(jsonBuf.data(), 1, jsonBuf.size(), fp);

			jsonBuf[jsonBuf.size() - 1] = '\0';

			const char *data = jsonBuf.data();
			if (jsonBuf.size() > 3 && (unsigned char)data[0] == 0xEF && (unsigned char)data[1] == 0xBB && (unsigned char)data[2] == 0xBF)
				data += 3;

			d.Parse(data);

			if (d.HasParseError())
				throw 0;

			for (auto it = d.MemberBegin(); it != d.MemberEnd(); ++it)
			{
				auto name = Utf8ToUtf16(it->name.GetString(), it->name.GetStringLength());
				auto val = Utf8ToUtf16(it->value.GetString(), it->value.GetStringLength());

				if(val.length() > 0)
					langStringMap.emplace(name, val);
			}
		}
		catch (...)
		{
		}

		if (fp)
			fclose(fp);
	}
		

public:
	void SetLangBaseDir(const tstring &LangBaseDir)
	{
		this->LangBaseDir = LangBaseDir;
		if (LangBaseDir.length() > 0 && (LangBaseDir.back() != TEXT('\\') && LangBaseDir.back() != TEXT('/')))
			this->LangBaseDir += TEXT('/');
	}

	bool ReadLangList(const tstring &LangListPath)
	{
		std::ifstream ifs(LangListPath);
		if (ifs.fail())
			return false;

		LangList.clear();

		bool isFirst = true;

		std::string str;
		while (getline(ifs, str))
		{
			if (isFirst && str.size() > 3 && (unsigned char)str[0] == 0xEF && (unsigned char)str[1] == 0xBB && (unsigned char)str[2] == 0xBF)
				str.erase(0, 3);

			isFirst = false;

			if (str.length() > 0 && str.front() == ';')
				continue;

			boost::char_separator<char> sep("\t");
			boost::tokenizer<boost::char_separator<char>> tokens(str, sep);

			std::vector<std::string> list;
			for (const auto& t : tokens)
				list.emplace_back(t);

			if (list.size() != 4)
				continue;

			stLangSetting ls;
			ls.LangName = Utf8ToUtf16(list[0].c_str(), list[0].length());
			ls.LangID = getNum(list[1]);
			ls.SubLangID = getNum(list[2]);
			ls.FileName = Utf8ToUtf16(list[3].c_str(), list[3].length());

			LangList.push_back(ls);
		}

		if (LangList.size() == 0)
			return false;

		if (NowLangID != 0)
			SetLang(NowLangID);

		// デフォルト言語を読みだす
		ReadLangFile(LangList[0], DefaultLangStringMap);

		return true;
	}

	void SetLang(const LANGID id)
	{
		NowLang = GetLang(id);
		NowLangID = id;

		ReadLangFile(NowLang, LangStringMap);
	}

	void SetLang(const stLangSetting &lang)
	{
		NowLang = lang;
		NowLangID = MAKELANGID(lang.LangID, lang.SubLangID);

		ReadLangFile(NowLang, LangStringMap);
	}

	void SetLang()
	{
		SetLang(GetUserDefaultUILanguage());
	}

	const stLangSetting& GetLang() const
	{
		return NowLang;
	}

	const std::vector<stLangSetting>& GetLangList() const
	{
		return LangList;
	}

	const std::wstring& GetString(const wchar_t *key) const
	{
		static std::wstring none;

		auto it = LangStringMap.find(key);
		if (it != LangStringMap.end())
			return it->second;

		it = DefaultLangStringMap.find(key);
		if (it != DefaultLangStringMap.end())
			return it->second;

		return none;
	}
};
