// ==================================================================
// Copyright (c) 2019 Alexander Freed. ALL RIGHTS RESERVED.
// Language: ISO C++14
//
// Useful functions.
// ==================================================================

#pragma once

#include <random>
#include <chrono>
#include <Eigen/Dense>


// global macros
#define UNUSED(x) ((void)x);  // for development only


namespace fnn {


/** A struct for holding globals
*/
struct Global
{
    /** get the random number generator
    @return a reference to the global URBG
    */
    static std::mt19937_64& rng()
    {
        // seed with the current time
        static std::mt19937_64 rand(get_seed());
        return rand; 
    }

    /** get the seed for the random number generator
    */
    static long long get_seed()
    {
        return get_seed_priv();
    }
    /** set the seed for the random number generator
    @param[in] seed The seed.
    */
    static void set_seed(const size_t seed)
    {
        get_seed_priv() = seed;
        rng().seed(seed);
    }

    /** set the seed for the random number generator to the default seed.
    */
    static void seed_default()
    {
        set_seed(std::mt19937_64::default_seed);
    }

private:
    /** manage the static seed instance.
    @return A non-const reference to the static seed instance.
    */
    static long long& get_seed_priv()
    {
        static long long seed = std::chrono::steady_clock::now().time_since_epoch().count();
        return seed;
    }
};


/** Eigen Maps cannot be swapped using operator=, as it overwrites the pointed-to array
@param[in/out] left  The map to swap with right
@param[in/out] right The map to swap with left
*/
template <typename T>
void SwapMap(Eigen::Map<T>& left, Eigen::Map<T>& right)
{
    Eigen::Map<T> temp = left;
    new (&left)  Eigen::Map<T>(&right(0, 0), right.rows(), right.cols());
    new (&right) Eigen::Map<T>(&temp(0, 0),  temp.rows(),  temp.cols());
}


}
