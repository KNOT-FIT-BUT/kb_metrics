#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2015 Brno University of Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Author: Matej Magdolen, xmagdo00@stud.fit.vutbr.cz
# Author: Jan Doležal, xdolez52@stud.fit.vutbr.cz
# Author: Lubomír Otrusina, iotrusina@fit.vutbr.cz
#
# Description: Loads a knowledge base and computes metrics and scores for static disambiguation.

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import re
import numpy
from enum import Enum
from orderedset import OrderedSet

# CONSTANTS
metrics_names = ["SCORE WIKI",	"SCORE METRICS", "CONFIDENCE"]
stats_names = ["WIKI BACKLINKS", "WIKI HITS", "WIKI PRIMARY SENSE"]
all_stats = stats_names + metrics_names

# getting the absolute path to the directory with this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_MULTIVALUE_DELIM = "|"

class KB_PART(Enum):
    HEAD = 1
    DATA = 2

# FUNCTIONS AND CLASSES
class KnowledgeBase:
    """
    * Pracuje s daty (sloupci) obsaženými na řádku v KB nebo v daném seznamu.
    * Sloupce jsou adresovány dle jejich názvù v HEAD_KB.
    * Tato třída je určena především pro přidání metrik do KB (metrics_to_KB.py), ale je používána pro svou jednoduchost i v dalších skriptech (kb_filter.py, prepare_kb_to_stats_and_metrics.py, wiki_stats_to_KB.py, KB2namelist.py).
    * Načítá KB až je to vyžadováno (lazy přístup). Je tedy možné načíst pouze zajímavé řádky z KB, převést je na seznamy a ty předávat místo čísel řádkù metodám instance této třídy, není-li nutno držet v pamìti celou KB.
    * Tisk instance této třídy vytiskne celou KB, byla-li načtena, jinak vytiskne pouze prázdný řetìzec.
    * Instance je reprezentována řetìzcem obsahujícím cestu k HEAD-KB, KB a dále informaci zda-li je KB načtena do pamìti.
    """

    def __init__(self, lang = "en", path_to_kb = None):
        if path_to_kb is None:
            self.path_to_kb = os.path.abspath(os.path.join(SCRIPT_DIR, "./inputs/KB_{}_all.tsv".format(lang)))
        else:
            self.path_to_kb = path_to_kb

        self.headKB, self.ent_type_col = self.getDictHeadKB(self.path_to_kb)

        self._kb_loaded = False
        self.lines = []

        # lists of metrics values in kb for computing percentiles
        self.metrics = {} # Slovník repr(set(ENTITY_TYPE_SET)):{str(METRIC):[int(VALUE), ...]}

        # data structure for indexing percentile scores
        self.metric_index = {} # Slovník repr(set(ENTITY_TYPE_SET)):{str(METRIC):{int(VALUE):float(PERCENTILE)}}

        self.multivalue_delim = "|"
        self.type_delim = "+"


    def __repr__(self):
        return "KnowledgeBase(path_to_kb=%r, kb_is_loaded=%r)" % (self.path_to_kb, self._kb_loaded)


    def getKBLines(self, fpath, kb_part):
        assert isinstance(kb_part, KB_PART)

        lines = []
        data_part = False
        with open(fpath) as fd:
            # skip version of KB
            next(fd)
            for line_num, line in enumerate(fd):
                if line == "\n":
                    if kb_part == KB_PART.HEAD:
                        break
                    data_part = True
                elif line != "":
                    if not data_part and kb_part == KB_PART.DATA:
                        continue
                    if data_part:
                        # Add missing line columns
                        line_split = line.split("\t")
                        line_cols = len(line_split)
                        ent_type = line_split[self.ent_type_col]
                        ent_type = ent_type.split(self.type_delim)
                        ent_type_set = OrderedSet(ent_type)
                        
                        if "__generic__" in self.headKB and "__generic__" not in ent_type_set:
                            ent_type_set = OrderedSet(["__generic__"]) | ent_type_set

                        min_line_cols = 0
                        for item in ent_type_set:
                            min_line_cols += len(self.headKB[item])  

                        # Add missing columns + add columns for possible stats                      
                        lines.append(line[:-1].split("\t") + ['' for _ in range(min_line_cols - line_cols + len(all_stats))])
                    else:
                        lines.append(line[:-1].split("\t"))
        return lines


    def getDictHeadKB(self, path_to_kb):
        """
        Returns a dictionary with the structure of KB from HEAD-KB and number of column with attribute TYPE.
        """

        PARSER_PATTERN = r"""
            (?:\{(?P<FLAGS>(?:\w|[ ])*)(?:\[(?P<PREFIX_OF_VALUE>[^\]]+)\])?\})?
            (?P<NAME>(?:\w|[ ])+)
        """

        PARSER_FIRST = re.compile(r"""(?ux)
            ^
            <(?P<TYPE>[^>]+)>
            (""" + PARSER_PATTERN + r""")?
            $
        """)

        PARSER_OTHER = re.compile(r"""(?ux)
            ^
            """ + PARSER_PATTERN + r"""
            $
        """)

        lines = self.getKBLines(self.path_to_kb, KB_PART.HEAD)

        headKB = {} # Slovník TYPE:{SUBTYPE:{COLUMN_NAME:COLUMN}}
        ent_type_col = None # Sloupec ve kterém je definován typ entity
        for line_num in range(len(lines)):
            plain_column = ""
            head_type = ""
            for col_num in range(len(lines[line_num])):
                plain_column = lines[line_num][col_num]
                if col_num == 0:
                    splitted = PARSER_FIRST.search(plain_column)
                    head_type = splitted.group("TYPE")
                    if head_type not in headKB:
                        headKB[head_type] = {}
                else:
                    splitted = PARSER_OTHER.search(plain_column)

                if splitted is not None: # This type has no defined columns
                    col_name = splitted.group("NAME")
                    headKB[head_type][col_name] = col_num

                if col_name == "TYPE":
                    if ent_type_col is None:
                        ent_type_col = col_num
                    elif ent_type_col != col_num:
                        raise RuntimeError("getDictHeadKB: TYPE column must be at same column for each type of entity in HEAD-KB!")
        return headKB, ent_type_col


    def check_or_load_kb(self):
        if not self._kb_loaded:
            self.load_kb()


    def load_kb(self):
        # loading knowledge base
        self.lines = self.getKBLines(self.path_to_kb, KB_PART.DATA)
        self._kb_loaded = True


    def get_ent_head(self, line):
        ent_type_set = self.get_ent_type(line)

        head = []
        for ent_supertype in ent_type_set:
            if ent_supertype not in self.headKB:
                raise Exception(f'Not defined type "{ent_supertype}" in head of KB.')
            head.extend([item[0] for item in sorted(self.headKB[ent_supertype].items(), key=lambda i: i[-1])])
        return head


    def get_ent_type(self, line):
        """ Returns a set of a type of an entity at the line of the knowledge base. """
        ent_type = self.get_field(line, self.ent_type_col)
        ent_type = ent_type.split(self.type_delim)
        ent_type_set = OrderedSet(ent_type)
        
        if "__generic__" in self.headKB and "__generic__" not in ent_type_set:
            ent_type_set = OrderedSet(["__generic__"]) | ent_type_set
        if "__stats__" in self.headKB and "__stats__" not in ent_type_set:
            ent_type_set = ent_type_set | OrderedSet(["__stats__"])
        
        return ent_type_set


    def get_location_code(self, line):
        return self.get_data_for(line, "FEATURE CODE")[0:3]


    def get_field(self, line, column):
        """ Returns a column of a line in the knowledge base. """
        try:
            if isinstance(line, list): # line jako sloupce dané entity
                return line[column]
            else: # line jako číslo řádku na kterém je daná entita
                self.check_or_load_kb()

                # KB lines are indexed from one
                return self.lines[int(line) - 1][column]
        except IndexError:
            raise RuntimeError("Line %s does not have column %s" % (line, column))
        except:
            raise


    def get_col_for(self, line, col_name, col_name_type=None):
        """ Line numbering from one. """

        # getting the entity type
        ent_type_set = self.get_ent_type(line)
        col = 0
        for type_name in ent_type_set:
            if self.headKB[type_name].get(col_name) is not None:
                return col + int(self.headKB[type_name][col_name])
            else:
                col += len(self.headKB[type_name])
        
        raise RuntimeError(f"Column name {col_name} does not exist for this line: {line}") 


    def get_data_for(self, line, col_name, col_name_type=None):
        """ Line numbering from one. """
        
        val = self.get_field(line, self.get_col_for(line, col_name, col_name_type))
        if val == '' or val == 'NF':
            val = '0'
        return val


    def nonempty_columns(self, line):
        """ Returns a number of columns at the specified line of the knowledge base which have a non-empty value. """

        if isinstance(line, list): # line jako sloupce dané entity
            columns = line
        else: # line jako číslo řádku na kterém je daná entita
            self.check_or_load_kb()
            columns = self.lines[line - 1]

        if "__stats__" in self.headKB:
            metrics_cols = [self.get_col_for(columns, colname, "__stats__") for colname in self.headKB["__stats__"].keys()]
        else:
            print("WARNING: No metrics columns was found => it will continue without metrics.", file=sys.stderr, flush=True)
            metrics_cols = []
        
        result = 0
        # KB lines are indexed from one
        for col in range(len(columns)):
            if col not in metrics_cols and columns[col]:
                result += 1

        return result


    def description_length(self, line):
        """ Returns a length of a description of a specified line. """

        return len(self.get_data_for(line, "DESCRIPTION"))

    def show_progress(self, val, max, interval=1, message=""):
        if val % interval == 0:
            print(f"{message}{(val/max)*100}%", end="\r")
        
        # Newline after finish
        if val == max:
            print()

    def metric_percentile(self, line, metric):
        """ Computing a percentile score for a given metric and entity. """

        # getting the entity type
        ent_type_set = self.get_ent_type(line)
        ent_type_set_index = repr(set(ent_type_set)) # using normal set for indexing because {1,3,2} == {1,2,3} but OrderedSet([1,3,2]) != OrderedSet([1,2,3])

        if metric == 'description_length':
            value = self.description_length(line)
        elif metric == 'columns_number':
            value = self.nonempty_columns(line)
        elif metric[0:4] == 'wiki':
            value_str = self.get_wiki_value(line, metric[5:])
            if value_str == "":
                value = 0
            else:
                value = int(value_str)
        return self.metric_index[ent_type_set_index][metric][value]


    def get_wiki_value(self, line, column_name):
        """
        Return a link to Wikipedia or a statistc value identified
        by column_name from knowledge base line.
        """

        column_rename = {'backlinks' : "WIKI BACKLINKS", 'hits' : "WIKI HITS", 'ps' : "WIKI PRIMARY SENSE"}
        if column_name == 'link':
            return self.get_data_for(line, "WIKIPEDIA URL")
        elif column_name in column_rename.keys():
            return self.get_data_for(line, column_rename[column_name])
        else:
            return self.get_data_for(line, column_name)

    # Check if all stats present in kb head
    def check_all_stats_present(self) -> bool:
        self.check_or_load_kb()
        stats = self.headKB['__stats__']
        for stat in all_stats:
            if stat not in stats.keys():
                return False
        return True

    # Checks head kb for '__stats__' line
    # if none found returns false
    # if found checks for backlinks, hits and ps (needed for calculating metrics)
    # adds empty columns for new metrics (needed in order for the script to run properly)
    def check_add_kb_stats(self) -> bool:
        # Check for neccessary staxs in KB head
        if "__stats__" not in self.headKB:
            return False

        stats = self.headKB['__stats__']

        # Check neccessary stats
        for stat in stats_names:
            if stat not in stats.keys():
                return False
        
        for metric in metrics_names:
            if metric not in stats.keys():
                self.headKB["__stats__"][metric] = len(self.headKB["__stats__"])
        
        return True
    
    # Inserts statistics (backlinks, pageviews and primary sense)
    # from an input file
    # Expected format <name> \t <backlinks> \t <pageviews> \t <primary sense>   
    def insert_stats(self, stats_file, save_changes=True) -> bool:
        # If stats already present, skip
        if "__stats__" in self.headKB:
            return False
        
        if not os.path.exists(stats_file):
            print("Stats file does not exist")
            return False
        
        # Add stats to kb head
        self.headKB['__stats__'] = {}
        for stat in stats_names:
            self.headKB["__stats__"][stat] = len(self.headKB["__stats__"])

        stats = self.headKB["__stats__"]

        print("Loading kb into memory")
        self.check_or_load_kb()
        
        # Load stats 
        print("Loading stats")
        stats = {}
        with open(stats_file, "r") as file_in:
            for line in file_in:
                values = line.split("\t")
                if len(values) != 4:
                    continue
                stats[values[0].replace("_"," ")] = [values[1], values[2], values[3].rstrip()]
        # Insert stats to KB
        # Add columns for stats
        print("Inserting stats")
        for line_num in range(1, len(self.lines) + 1):
            columns = self.lines[line_num - 1]     
            art_name = columns[self.get_col_for(columns, "NAME")]
            if stats.get(art_name):
                val = stats.pop(art_name)
                for idx, stat in enumerate(stats_names):
                    columns[self.get_col_for(columns, stat)] = val[idx]
            self.lines[line_num - 1] = columns
        
        del stats

        # Insert metrics (calculated from the 3 stats)
        self.insert_metrics()

        if save_changes:
            self.save_changes()
      

    # Computing SCORE WIKI, SCORE METRICS and CONFIDENCE and adding them to the KB
    def insert_metrics(self, save_changes=True):
        
        self.check_or_load_kb()

        # CHECK INPUT KB:
        # If all stats and metrics are already present
        if self.check_all_stats_present():
            print("Warning: NO METRICS ADDED!")
            print("-> all stats and metrics already present")
            return

        # If some neccessary stats missing
        if not self.check_add_kb_stats():
            print("Warning: NO METRICS ADDED!")
            print("-> missing neccessary stats (backlinks/hits/ps)")
            return

        
        # computing statistics
        for line_num in range(1, len(self.lines) + 1):
            self.show_progress(line_num, len(self.lines), interval=10000, message="Computing statistics... ")
            ent_type_set = self.get_ent_type(line_num)
            ent_type_set_index = repr(set(ent_type_set)) # using normal set for indexing because {1,3,2} == {1,2,3} but OrderedSet([1,3,2]) != OrderedSet([1,2,3])
            self.metrics.setdefault(ent_type_set_index, {})
            self.metrics[ent_type_set_index].setdefault('columns_number', []).append(self.nonempty_columns(line_num))
            self.metrics[ent_type_set_index].setdefault('description_length', []).append(self.description_length(line_num))
            if self.get_wiki_value(line_num, 'backlinks'):
                self.metrics[ent_type_set_index].setdefault('wiki_backlinks', []).append(int(self.get_wiki_value(line_num, 'backlinks')))
                self.metrics[ent_type_set_index].setdefault('wiki_hits', []).append(int(self.get_wiki_value(line_num, 'hits')))
                self.metrics[ent_type_set_index].setdefault('wiki_ps', []).append(int(self.get_wiki_value(line_num, 'ps')))

        # sorting statistics
        for i in self.metrics:
            for j in self.metrics[i]:
                self.metrics[i][j].sort()

        # indexing statistics
        for i in self.metrics:
            for j in self.metrics[i]:
                for k in range(0, len(self.metrics[i][j])):
                    self.show_progress(line_num, len(self.lines), interval=10000, message="Indexing statistics... ")
                    if self.metrics[i][j][k] not in self.metric_index.setdefault(i, {}).setdefault(j, {}):
                        max_value = float(self.metrics[i][j][-1])
                        if j in ['wiki_backlinks', 'wiki_hits']:
                            max_value = 0.25 * max_value
                        if max_value:
                            normalized_value = float(self.metrics[i][j][k]) / max_value
                            self.metric_index[i][j][self.metrics[i][j][k]] = min(normalized_value, 1.0)
                        else:
                            self.metric_index[i][j][self.metrics[i][j][k]] = 1.0

        # computing SCORE WIKI, SCORE METRICS and CONFIDENCE
        for line_num in range(1, len(self.lines) + 1):
            self.show_progress(line_num, len(self.lines), interval=10000, message="Computing metrics... ")

            columns = self.lines[line_num - 1]
            
            # computing SCORE WIKI
            score_wiki = 0
            if self.get_wiki_value(columns, 'backlinks'):
                wiki_backlinks = self.metric_percentile(columns, 'wiki_backlinks')
                wiki_hits = self.metric_percentile(columns, 'wiki_hits')
                wiki_ps = self.metric_percentile(columns, 'wiki_ps')
                score_wiki = 100 * numpy.average([wiki_backlinks, wiki_hits, wiki_ps], weights=[5, 5, 1])
            columns[self.get_col_for(columns, "SCORE WIKI")] = "%.2f" % score_wiki

            # computing SCORE METRICS
            description_length = self.metric_percentile(columns, 'description_length')
            columns_number = self.metric_percentile(columns, 'columns_number')
            score_metrics = 100 * numpy.average([description_length, columns_number])
            columns[self.get_col_for(columns, "SCORE METRICS")] = "%.2f" % score_metrics

            # computing CONFIDENCE
            columns[self.get_col_for(columns, "CONFIDENCE")] = "%.2f" % numpy.average([score_wiki, score_metrics], weights=[5, 1])
        
        if save_changes:
            self.save_changes()
    
    def save_changes(self, output_file=""):
        # Add '+stats' to filename
        if not output_file:
            file_path = os.path.dirname(os.path.abspath(self.path_to_kb))
            file_name = os.path.basename(os.path.abspath(self.path_to_kb))
            file_extension = file_name.split(".")[-1]
            file_name = "".join(file_name.split(".")[:-1])
            if not file_extension:
                file_extension = "tsv"
            
            file_name += "+stats"
            output_file = f"{file_path}/{file_name}.{file_extension}"
        else:
            output_file = self.path_to_kb

        print("saving changes to ", output_file)
        with open(output_file, "w") as out_file:
            # Save KB head
            KB_lines = self.getKBLines(self.path_to_kb, KB_PART.HEAD)
            stats_added = False
            for line in KB_lines:
                # Add new columns in __stats__ line (if new metrics were inserted)
                if any("<__stats__>" in column for column in line):
                    out_file.write("<__stats__>" + "\t".join(self.headKB["__stats__"].keys()))
                    stats_added = True
                else:
                    out_file.write("\t".join(line))
                out_file.write("\n")
            if not stats_added:
                out_file.write("<__stats__>" + "\t".join(self.headKB["__stats__"].keys()))
                out_file.write("\n")


            # head-data separator
            out_file.write("\n")

            # Save KB data
            for line in self.lines+[""]:
                out_file.write("\t".join(line))
                out_file.write("\n")
        
    def _str1(self):
        return '\n'.join(['\t'.join(line) for line in self.lines+[""]])

    def _str2(self):
        result = ""
        for line in self.lines:
            result += '\t'.join(line) + '\n'
        return result

    def __str__(self):
        return self._str1()

